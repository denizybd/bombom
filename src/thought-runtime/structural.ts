import { clamp } from './math'
import type { ThoughtBranch, ThoughtBranchScore, ThoughtEpisodeState } from './types'

function scoreBranch(branch: ThoughtBranch): ThoughtBranchScore {
  const count = branch.transitions.length
  if (!count) return {
    branchId: branch.id,
    transitionCount: 0,
    meanCompatibility: 0,
    meanGraphSupport: 0,
    meanFamilySupport: 0,
    uniqueTransitionRatio: 0,
    loopRatio: 1,
    score: 0,
    eligible: false,
    explanation: 'Geçiş yok; yapısal kanıt üretilemedi.',
  }
  const meanCompatibility = branch.transitions.reduce((sum, transition) => sum + transition.compatibility, 0) / count
  const meanGraphSupport = branch.transitions.reduce((sum, transition) => sum + transition.habitProbability, 0) / count
  const meanFamilySupport = branch.transitions.reduce((sum, transition) => sum + transition.membershipWeight, 0) / count
  const keys = branch.transitions.map((transition) => `${transition.fromNeuronId}\u0000${transition.toNeuronId}`)
  const uniqueTransitionRatio = new Set(keys).size / count
  const loopRatio = 1 - new Set(branch.path).size / Math.max(1, branch.path.length)
  const score = clamp(
    0.4 * meanCompatibility + 0.2 * meanGraphSupport + 0.2 * meanFamilySupport +
    0.2 * uniqueTransitionRatio - 0.35 * loopRatio,
  )
  const eligible = count >= 2 && score >= 0.5 && loopRatio <= 0.5
  return {
    branchId: branch.id,
    transitionCount: count,
    meanCompatibility,
    meanGraphSupport,
    meanFamilySupport,
    uniqueTransitionRatio,
    loopRatio,
    score,
    eligible,
    explanation: eligible
      ? 'Yeterli geçiş, uyum ve çeşitlilik var; sonuç kapısına aday.'
      : 'Yapısal skor yalnız diagnostic; tek başına kalıcı öğrenme kredisi vermez.',
  }
}

export function scoreThoughtBranches(episode: ThoughtEpisodeState): ThoughtBranchScore[] {
  return episode.branches.map(scoreBranch)
    .sort((left, right) => right.score - left.score || left.branchId.localeCompare(right.branchId))
}
