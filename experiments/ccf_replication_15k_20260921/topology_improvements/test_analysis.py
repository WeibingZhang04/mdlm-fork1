import unittest,math
from analyze_results import paired_comparison
class AnalysisTest(unittest.TestCase):
    def row(self,i,nll,tokens):
        return {'pair_key':str(i),'pair_seed':123+i,'reference_lm':{'mean_nll_nats':nll,'token_count':tokens,'model_name_or_path':'fixed','revision':'fixed','sequence_policy':'same'}}
    def test_token_weighting_and_identity(self):
        a=[self.row(0,1,1),self.row(1,3,9)];b=[self.row(0,1,1),self.row(1,1,9)]
        out=paired_comparison(a,b,100)
        self.assertAlmostEqual(out['delta_mean_nll'],1.8)
        self.assertAlmostEqual(out['ppl_ratio'],math.exp(1.8))
        same=paired_comparison(a,a,100)
        self.assertEqual(same['ppl_ratio_ci95'],[1.,1.])
    def test_no_silent_sample_or_seed_dropping(self):
        a=[self.row(0,1,1),self.row(1,3,9)]
        with self.assertRaises(ValueError):paired_comparison(a,a[:1],100)
        b=[self.row(0,1,1),self.row(1,3,9)];b[1]['pair_seed']+=1
        with self.assertRaises(ValueError):paired_comparison(a,b,100)
if __name__=='__main__':unittest.main()
