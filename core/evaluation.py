# evaluation.py
class Evaluation:
    def rank_steps(self, results):
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def generate_insights(self, results):
        return [result['insight'] for result in results if 'insight' in result]

    def generate_summary(self, results):
        return "\n".join([result['summary'] for result in results if 'summary' in result])