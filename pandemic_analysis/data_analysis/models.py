from django.db import models

class PandemicData(models.Model):
    date = models.DateField(null=True)
    totalCases = models.IntegerField(null=True)
    newCases = models.IntegerField(null=True)
    intenciveCareUnit = models.IntegerField(null=True)
    deaths = models.IntegerField(null=True)
    newDeaths = models.IntegerField(null=True)
    totalTests = models.IntegerField(null=True)
    PCRperDay = models.IntegerField(null=True)

def __str__(self):
    formatted_date = self.date.strftime('%d/%m/%y')
    return f"Date: {formatted_date }, Total Cases: {self.totalCases}, New Cases: {self.newCases}, Deaths: {self.deaths}, PCR Per Day: {self.PCRperDay}"
