import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { defaultConfig } from '../config.mjs'
import { isPrivateAddress, readAllowedFile } from '../source-policy.mjs'
import { temporaryDirectory } from './helpers.mjs'

test('file ingestion resolves canonical paths and blocks traversal', async (t) => {
  const root = await temporaryDirectory(t)
  const watched = path.join(root, 'watched')
  await mkdir(watched)
  const allowedFile = path.join(watched, 'book.txt')
  const outsideFile = path.join(root, 'secret.txt')
  await writeFile(allowedFile, '  Kontrollü\reğitim I\u0307stanbul.  ')
  await writeFile(outsideFile, 'Kapsam dışı metin.')
  const config = defaultConfig(root)
  config.sourcePolicy.watchedDirectories = [watched]
  const loaded = await readAllowedFile(allowedFile, config)
  assert.equal(loaded.sourceType, 'local-file')
  assert.equal(loaded.content, 'Kontrollü\neğitim İstanbul.')
  await assert.rejects(readAllowedFile(outsideFile, config), { code: 'PATH_OUTSIDE_ALLOWLIST' })
})

test('URL policy rejects private IPv4 addresses hidden inside IPv6 mappings', () => {
  for (const address of [
    '::ffff:172.16.0.1',
    '::ffff:169.254.0.1',
    '::ffff:ac10:1',
    '0:0:0:0:0:ffff:a9fe:1',
    '::ffff:0:ac10:1',
    '64:ff9b::a9fe:1',
    '::127.0.0.1',
  ]) assert.equal(isPrivateAddress(address), true, address)

  assert.equal(isPrivateAddress('::ffff:8.8.8.8'), false)
  assert.equal(isPrivateAddress('2606:4700:4700::1111'), false)
})
