import { createHash } from 'node:crypto'
import { readFile, realpath, stat } from 'node:fs/promises'
import path from 'node:path'
import dns from 'node:dns/promises'
import net from 'node:net'
import http from 'node:http'
import https from 'node:https'

export const sha256 = (content) => createHash('sha256').update(content).digest('hex')
export const byteLength = (content) => Buffer.byteLength(content, 'utf8')

function policyError(message, code = 'SOURCE_POLICY_REJECTED') {
  const error = new Error(message)
  error.code = code
  return error
}

export function normalizeTextSource({ text, title = 'Elle eklenen metin', language = 'tr', provenance = {}, preserveWhitespace = false }, config) {
  if (typeof text !== 'string') throw policyError('Kaynak metni bir string olmalı.', 'INVALID_SOURCE')
  const canonical = text.replace(/\r\n?/g, '\n').normalize('NFC')
  const content = preserveWhitespace ? canonical : canonical.trim()
  const bytes = byteLength(content)
  if (!content.trim()) throw policyError('Boş metin kaynak olarak eklenemez.', 'INVALID_SOURCE')
  if (bytes > config.budgets.maxSourceBytes) throw policyError('Metin kaynak boyutu sınırını aşıyor.', 'SOURCE_TOO_LARGE')
  return {
    title: String(title).trim().slice(0, 240) || 'Adsız metin',
    sourceType: 'manual-text',
    locator: null,
    content,
    contentHash: sha256(content),
    bytes,
    language: String(language || 'tr').slice(0, 16),
    provenance: { kind: 'manual', receivedAt: new Date().toISOString(), ...safeProvenance(provenance) },
  }
}

export async function readAllowedFile(filePath, config) {
  if (typeof filePath !== 'string' || filePath.includes('\0')) throw policyError('Geçersiz dosya yolu.', 'INVALID_PATH')
  if (!config.sourcePolicy.watchedDirectories.length) throw policyError('Önce en az bir izinli kaynak klasörü tanımlanmalı.', 'NO_WATCHED_DIRECTORY')
  const resolvedFile = await realpath(path.resolve(filePath))
  const roots = await Promise.all(config.sourcePolicy.watchedDirectories.map(async (root) => realpath(path.resolve(root))))
  const allowed = roots.some((root) => {
    const relative = path.relative(root, resolvedFile)
    return relative !== '' ? !relative.startsWith('..') && !path.isAbsolute(relative) : false
  })
  if (!allowed) throw policyError('Dosya izinli kaynak klasörlerinin dışında.', 'PATH_OUTSIDE_ALLOWLIST')
  const extension = path.extname(resolvedFile).toLowerCase()
  if (!config.sourcePolicy.allowedExtensions.includes(extension)) throw policyError(`.${extension} uzantısına izin verilmiyor.`, 'EXTENSION_NOT_ALLOWED')
  const info = await stat(resolvedFile)
  if (!info.isFile()) throw policyError('Kaynak yolu normal bir dosya değil.', 'INVALID_SOURCE_FILE')
  if (info.size > config.budgets.maxSourceBytes) throw policyError('Dosya kaynak boyutu sınırını aşıyor.', 'SOURCE_TOO_LARGE')
  const content = (await readFile(resolvedFile, 'utf8')).replace(/\r\n?/g, '\n').normalize('NFC').trim()
  if (!content) throw policyError('Boş dosya kaynak olarak eklenemez.', 'INVALID_SOURCE')
  return {
    title: path.basename(resolvedFile),
    sourceType: 'local-file',
    locator: resolvedFile,
    content,
    contentHash: sha256(content),
    bytes: byteLength(content),
    language: 'tr',
    provenance: { kind: 'local-file', canonicalPath: resolvedFile, modifiedAt: info.mtime.toISOString(), receivedAt: new Date().toISOString() },
  }
}

const blockedIpv4 = new net.BlockList()
for (const [network, prefix] of [
  ['0.0.0.0', 8],
  ['10.0.0.0', 8],
  ['100.64.0.0', 10],
  ['127.0.0.0', 8],
  ['169.254.0.0', 16],
  ['172.16.0.0', 12],
  ['192.0.0.0', 24],
  ['192.0.2.0', 24],
  ['192.88.99.0', 24],
  ['192.168.0.0', 16],
  ['198.18.0.0', 15],
  ['198.51.100.0', 24],
  ['203.0.113.0', 24],
  ['224.0.0.0', 3],
]) blockedIpv4.addSubnet(network, prefix, 'ipv4')

const blockedIpv6 = new net.BlockList()
for (const [network, prefix] of [
  ['::', 128],
  ['::1', 128],
  ['100::', 64],
  ['2001::', 32],
  ['2001:db8::', 32],
  ['2002::', 16],
  ['64:ff9b:1::', 48],
  ['fc00::', 7],
  ['fe80::', 10],
  ['ff00::', 8],
]) blockedIpv6.addSubnet(network, prefix, 'ipv6')

function parseIpv6Words(address) {
  let literal = address.toLowerCase().split('%', 1)[0]
  const dottedTail = literal.match(/(?:^|:)(\d{1,3}(?:\.\d{1,3}){3})$/)?.[1]
  if (dottedTail) {
    if (!net.isIPv4(dottedTail)) return null
    const octets = dottedTail.split('.').map(Number)
    const encoded = `${((octets[0] << 8) | octets[1]).toString(16)}:${((octets[2] << 8) | octets[3]).toString(16)}`
    literal = `${literal.slice(0, -dottedTail.length)}${encoded}`
  }

  const halves = literal.split('::')
  if (halves.length > 2) return null
  const left = halves[0] ? halves[0].split(':') : []
  const right = halves.length === 2 && halves[1] ? halves[1].split(':') : []
  const missing = 8 - left.length - right.length
  if ((halves.length === 1 && missing !== 0) || missing < 0) return null
  const words = [...left, ...Array(missing).fill('0'), ...right]
  if (words.length !== 8 || words.some((word) => !/^[0-9a-f]{1,4}$/.test(word))) return null
  return words.map((word) => Number.parseInt(word, 16))
}

function mappedIpv4(address) {
  const words = parseIpv6Words(address)
  if (!words) return null
  const standardMapped = words.slice(0, 5).every((word) => word === 0) && words[5] === 0xffff
  const translatedMapped = words.slice(0, 4).every((word) => word === 0) && words[4] === 0xffff && words[5] === 0
  const compatibleMapped = words.slice(0, 6).every((word) => word === 0) && (words[6] !== 0 || words[7] > 1)
  const wellKnownNat64 = words[0] === 0x64 && words[1] === 0xff9b && words.slice(2, 6).every((word) => word === 0)
  if (!standardMapped && !translatedMapped && !compatibleMapped && !wellKnownNat64) return null
  return `${words[6] >>> 8}.${words[6] & 0xff}.${words[7] >>> 8}.${words[7] & 0xff}`
}

export function isPrivateAddress(address) {
  if (net.isIPv4(address)) {
    return blockedIpv4.check(address, 'ipv4')
  }
  if (net.isIPv6(address)) {
    const embedded = mappedIpv4(address)
    return embedded ? isPrivateAddress(embedded) : blockedIpv6.check(address.split('%', 1)[0], 'ipv6')
  }
  return true
}

async function assertPublicAllowlistedUrl(url, config) {
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw policyError('Yalnızca kimlik bilgisiz HTTP(S) kaynakları kabul edilir.', 'URL_NOT_ALLOWED')
  if (!config.sourcePolicy.allowedHttpOrigins.includes(url.origin)) throw policyError('Adres açık HTTP kaynak izin listesinde değil.', 'URL_NOT_ALLOWLISTED')
  const addresses = await dns.lookup(url.hostname, { all: true, verbatim: true })
  if (!addresses.length || addresses.some(({ address }) => isPrivateAddress(address))) {
    throw policyError('Yerel veya özel ağ adreslerinden kaynak alınamaz.', 'PRIVATE_ADDRESS_BLOCKED')
  }
  return addresses
}

export async function fetchAllowedUrl(inputUrl, config, { signal } = {}) {
  let url = new URL(inputUrl)
  for (let redirect = 0; redirect <= 3; redirect += 1) {
    const addresses = await assertPublicAllowlistedUrl(url, config)
    const response = await requestPinnedUrl(url, addresses, config.budgets.maxSourceBytes, signal)
    if ([301, 302, 303, 307, 308].includes(response.status)) {
      if (redirect === 3) throw policyError('Kaynak çok fazla yönlendirme yaptı.', 'TOO_MANY_REDIRECTS')
      const location = response.headers.location
      if (!location) throw policyError('Kaynak yönlendirmesi geçersiz.', 'INVALID_REDIRECT')
      url = new URL(location, url)
      continue
    }
    if (response.status < 200 || response.status >= 300) throw policyError(`Kaynak HTTP ${response.status} döndürdü.`, 'SOURCE_FETCH_FAILED')
    const content = new TextDecoder('utf-8', { fatal: true }).decode(response.body).replace(/\r\n?/g, '\n').normalize('NFC').trim()
    if (!content) throw policyError('Uzak kaynak boş.', 'INVALID_SOURCE')
    return {
      title: url.pathname.split('/').filter(Boolean).at(-1)?.slice(0, 240) || url.hostname,
      sourceType: 'allowlisted-url', locator: url.toString(), content,
      contentHash: sha256(content), bytes: byteLength(content), language: 'tr',
      provenance: { kind: 'allowlisted-url', url: url.toString(), receivedAt: new Date().toISOString() },
    }
  }
  throw policyError('Kaynak alınamadı.', 'SOURCE_FETCH_FAILED')
}

function requestPinnedUrl(url, addresses, maxBytes, signal) {
  return new Promise((resolve, reject) => {
    const transport = url.protocol === 'https:' ? https : http
    const lookup = (_hostname, options, callback) => {
      const requestedFamily = typeof options === 'number' ? options : options?.family
      const candidates = requestedFamily ? addresses.filter((item) => item.family === requestedFamily) : addresses
      const selected = candidates[0] ?? addresses[0]
      if (typeof options === 'object' && options?.all) callback(null, candidates.length ? candidates : addresses)
      else callback(null, selected.address, selected.family)
    }
    const request = transport.request(url, {
      method: 'GET', lookup,
      headers: { Accept: 'text/plain, text/markdown, application/json;q=0.8', 'User-Agent': 'Bombom-Local-Training/1.0' },
    }, (response) => {
      const status = response.statusCode ?? 0
      const length = Number(response.headers['content-length'] ?? 0)
      if (length > maxBytes) {
        response.destroy()
        reject(policyError('Uzak kaynak boyutu sınırı aşıyor.', 'SOURCE_TOO_LARGE'))
        return
      }
      const chunks = []
      let total = 0
      response.on('data', (chunk) => {
        total += chunk.length
        if (total > maxBytes) response.destroy(policyError('Uzak kaynak boyutu sınırı aşıyor.', 'SOURCE_TOO_LARGE'))
        else chunks.push(chunk)
      })
      response.on('end', () => resolve({ status, headers: response.headers, body: Buffer.concat(chunks) }))
      response.on('error', reject)
    })
    request.on('error', reject)
    if (signal) {
      if (signal.aborted) request.destroy(signal.reason)
      else signal.addEventListener('abort', () => request.destroy(signal.reason), { once: true })
    }
    request.end()
  })
}

function safeProvenance(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const result = {}
  for (const [key, item] of Object.entries(value).slice(0, 32)) {
    if (typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean' || item === null) result[String(key).slice(0, 80)] = typeof item === 'string' ? item.slice(0, 1_000) : item
  }
  return result
}
