from collections import Counter
import math
from typing import List

class BLEU:
    def __init__(self, n=4):
        self.n = n

    def _get_ngrams(self, tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def get_score(self, cand: List[List[int]], ref: List[List[int]]):

        
        cand = [[item for item in lst if item != 0] for lst in cand] 
        ref = [[item for item in lst if item != 0] for lst in ref] 

        
        total_clipped = [0] * self.n
        total_counts  = [0] * self.n

        
        for candidate, reference in zip(cand, ref):
            for i in range(self.n):
                n = i + 1

                cand_ngrams = self._get_ngrams(candidate, n)
                ref_ngrams  = self._get_ngrams(reference, n)

                cand_counts = Counter(cand_ngrams)
                ref_counts  = Counter(ref_ngrams)
                
                clipped = 0
                for ng in cand_counts:
                    clipped += min(cand_counts[ng], ref_counts.get(ng, 0))

                total_clipped[i] += clipped
                total_counts[i]  += sum(cand_counts.values())

        log_precisions = []

        for i in range(self.n):
            if total_counts[i] == 0:
                continue

            if total_clipped[i] == 0:
                p_i = 1 / (total_counts[i] * 2)
            else:
                p_i = total_clipped[i] / total_counts[i]

            log_precisions.append(math.log(p_i))

        if len(log_precisions) == 0:
            return 0.0

        bleu = math.exp(sum(log_precisions) / len(log_precisions))

        c = sum(len(c) for c in cand)
        r = sum(len(r) for r in ref)

        if c > r:
            BP = 1
        else:
            BP = math.exp(1 - r / c) if c > 0 else 0

        return BP * bleu
        
if __name__ == "__main__":
    
    scoring = BLEU(2)
    print("scoring")
    candidate = [[1,2,3,4]]
    reference = [[1,2,2,2,3,4]]

    print(scoring.get_score(candidate,reference))
    

    