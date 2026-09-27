from django.test import TestCase


class HomePageTests(TestCase):
    def test_home_page_includes_evaluation_controls(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="evaluationModelSelect"')
        self.assertContains(response, 'id="evaluationFeatureSelect"')
        self.assertContains(response, 'id="evaluationBtn"')
        self.assertContains(response, 'id="evaluationResultContainer"')
        self.assertContains(response, 'fetch("/api/evaluation/"')
        self.assertContains(response, 'id="plotModelSelect"')
        self.assertContains(response, 'id="plotFeatureSelect"')
        self.assertContains(response, 'id="plotBtn"')
        self.assertContains(response, 'id="plotImage"')
        self.assertContains(response, 'fetch("/api/plot/"')
        self.assertContains(response, 'window.location.protocol === "file:"')
        self.assertContains(response, 'window.location.replace("http://127.0.0.1:8000/"')
