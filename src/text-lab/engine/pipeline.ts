import { createMatrixLayout, MAX_ANALYSIS_NODES, validateConfig } from '../domain/config'
import type {
  PipelineProgress,
  RawTextNeuron,
  TextLabAnalysis,
  TextLabConfig,
  TextSource,
} from '../domain/types'
import { propagateContext } from './context'
import { encodeIdentity } from './encode'
import { vectorNorm } from './matrix'
import {
  createContextSpaceId,
  createIdentitySpaceId,
  createRepresentationSpaceId,
  digestText,
  digestVector,
} from './representation'
import { buildSpanForest, canonicalizeSource, countGraphemes } from './segment'
import {
  computeAllPairSimilarities,
  computeInitialHabitProbabilitiesAsync,
  PipelineCancelledError,
  type SimilarityOptions,
} from './similarity'

export interface PipelineOptions extends SimilarityOptions {}

function emit(
  options: PipelineOptions,
  stage: PipelineProgress['stage'],
  completed: number,
  total: number,
  message: string,
) {
  const localRatio = total === 0 ? 1 : completed / total
  const ranges: Record<PipelineProgress['stage'], readonly [number, number]> = {
    preparing: [0, 0.03],
    segmenting: [0.03, 0.12],
    encoding: [0.12, 0.32],
    propagating: [0.32, 0.4],
    comparing: [0.4, 0.93],
    connecting: [0.93, 0.99],
    complete: [1, 1],
  }
  const [start, end] = ranges[stage]
  options.onProgress?.({
    stage,
    completed,
    total,
    ratio: start + (end - start) * localRatio,
    message,
  })
}

function throwIfCancelled(options: PipelineOptions): void {
  if (options.shouldCancel?.()) throw new PipelineCancelledError()
}

export async function analyzeText(
  sourceInput: TextSource,
  config: TextLabConfig,
  options: PipelineOptions = {},
): Promise<TextLabAnalysis> {
  const startedAt = performance.now()
  validateConfig(config)
  emit(options, 'preparing', 0, 1, 'Kaynak kayıpsız biçimde hazırlanıyor…')
  const source = canonicalizeSource(sourceInput)
  if (!source.text.trim()) throw new RangeError('Analiz için boş olmayan bir kaynak metin gerekli.')
  const layout = createMatrixLayout(config)
  const sourceDigest = digestText(source.text)
  const identitySpaceId = createIdentitySpaceId(config, layout)
  const contextSpaceId = createContextSpaceId(config, layout)
  const representationSpaceId = createRepresentationSpaceId(identitySpaceId, contextSpaceId)
  emit(options, 'preparing', 1, 1, 'Kaynak hazır')
  throwIfCancelled(options)

  emit(options, 'segmenting', 0, 1, 'Paragraflar bölünme ağacına dönüştürülüyor…')
  const forest = buildSpanForest(
    source,
    config.minimumLeafGraphemes,
    MAX_ANALYSIS_NODES,
  )
  if (!forest.nodes.length) throw new RangeError('Kaynakta işlenebilir bir paragraf bulunamadı.')
  emit(options, 'segmenting', 1, 1, `${forest.nodes.length.toLocaleString('tr-TR')} ham nöron oluşturuldu`)
  throwIfCancelled(options)

  const nodes: RawTextNeuron[] = []
  for (let index = 0; index < forest.nodes.length; index += 1) {
    const draft = forest.nodes[index]
    const identity = encodeIdentity(draft.text, layout, config)
    nodes.push({
      ...draft,
      identity,
      relation: new Float32Array(layout.relationLength),
      identityDigest: digestVector(identity),
      surfaceKey: `SF-${digestText(draft.text).toUpperCase()}`,
      identityNorm: vectorNorm(identity),
      relationNorm: 0,
      contextSupport: 0,
      contextRevision: 0,
      trace: {
        inheritedNorm: 0,
        parentIdentityCompatibility: 0,
        parentGain: 0,
        siblingId: null,
        siblingIdentityCompatibility: null,
        siblingGain: 0,
      },
    })
    if (index % 128 === 0 || index === forest.nodes.length - 1) {
      emit(options, 'encoding', index + 1, forest.nodes.length, 'Yüzey kimlikleri koordinatlara yazılıyor…')
      if (index > 0) {
        await new Promise<void>((resolve) => setTimeout(resolve, 0))
        throwIfCancelled(options)
      }
    }
  }

  emit(options, 'propagating', 0, nodes.length, 'Üst bağlam ve kardeş mikro-izleri aktarılıyor…')
  propagateContext(nodes, forest.rootIds, layout, config)
  if (nodes.some((node) => digestVector(node.identity) !== node.identityDigest)) {
    throw new TypeError('Bağlam üretimi sabit kimlik alanlarından birini değiştirdi.')
  }
  emit(options, 'propagating', nodes.length, nodes.length, 'Bağlamsal ilişki alanları hazır')
  await new Promise<void>((resolve) => setTimeout(resolve, 0))
  throwIfCancelled(options)

  const similarityResult = await computeAllPairSimilarities(nodes, layout, config, {
    ...options,
    onProgress: (progress) => {
      const mappedRatio = 0.4 + 0.53 * progress.ratio
      options.onProgress?.({ ...progress, ratio: mappedRatio })
    },
  })
  await new Promise<void>((resolve) => setTimeout(resolve, 0))
  throwIfCancelled(options)
  emit(options, 'connecting', 0, nodes.length, 'Pozitif benzerlik kanıtı geçiş olasılıklarına çevriliyor…')
  const habitProbabilities = await computeInitialHabitProbabilitiesAsync(
    similarityResult.similarities,
    nodes,
    config.habitExponent,
    {
      shouldCancel: options.shouldCancel,
      onProgress: (completed, total) =>
        emit(
          options,
          'connecting',
          completed,
          total,
          completed <= nodes.length
            ? 'Pozitif K kanıtları hazırlanıyor…'
            : 'Başlangıç H⁰ satırları normalize ediliyor…',
        ),
    },
  )
  emit(options, 'connecting', nodes.length, nodes.length, 'Başlangıç bağlantı olasılıkları hazır')

  const durationMs = performance.now() - startedAt
  const uniquePairCount = (nodes.length * (nodes.length - 1)) / 2
  const analysis: TextLabAnalysis = {
    format: 'metin-laboratuvari-analysis',
    version: 2,
    runId: `RUN-${digestText(`${source.id}:${sourceDigest}:${representationSpaceId}:${config.minimumLeafGraphemes}`).toUpperCase()}`,
    createdAt: new Date().toISOString(),
    sourceDigest,
    identitySpaceId,
    contextSpaceId,
    representationSpaceId,
    source,
    config: { ...config },
    layout,
    rootIds: forest.rootIds,
    nodes,
    similarities: similarityResult.similarities,
    habitProbabilities,
    stats: {
      characterCount: source.text.length,
      graphemeCount: countGraphemes(source.text),
      paragraphCount: forest.rootIds.length,
      neuronCount: nodes.length,
      uniquePairCount,
      matrixBytes: nodes.reduce(
        (bytes, node) => bytes + node.identity.byteLength + node.relation.byteLength,
        0,
      ),
      similarityBytes:
        similarityResult.similarities.byteLength + habitProbabilities.byteLength,
      durationMs,
      medianRelationNorm: similarityResult.medianRelationNorm,
    },
  }
  emit(options, 'complete', 1, 1, 'Metin Laboratuvarı deneyi tamamlandı')
  return analysis
}
