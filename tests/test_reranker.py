import unittest
from src.reranker import DocumentReranker

class TestReranker(unittest.TestCase):
    def setUp(self):
        self.reranker = DocumentReranker()

    def test_reranking_accuracy(self):
        candidates = [
            {
                "id": "c1",
                "text": "The SOME/IP protocol is used for service-oriented communication over Ethernet.",
                "metadata": {"source_file": "SWS_SOMEIP.pdf", "page_number": 10},
                "score": 0.5
            },
            {
                "id": "c2",
                "text": "The CAN Driver controls low-level hardware registers of the CAN controller.",
                "metadata": {"source_file": "SWS_CAN.pdf", "page_number": 5},
                "score": 0.6
            },
            {
                "id": "c3",
                "text": "LIN Driver operates with master-slave scheduling frames.",
                "metadata": {"source_file": "SWS_LIN.pdf", "page_number": 8},
                "score": 0.4
            }
        ]

        query = "What protocol is used for service communication over Ethernet?"
        reranked = self.reranker.rerank(query=query, candidate_chunks=candidates, top_k=2)

        self.assertEqual(len(reranked), 2)
        # Verify the SOME/IP Ethernet chunk is elevated to rank #1
        self.assertEqual(reranked[0]["id"], "c1")
        self.assertIn("rerank_score", reranked[0])

if __name__ == "__main__":
    unittest.main()
