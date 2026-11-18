from app.analytics.statistics.descriptive_statistics import (
    DescriptiveStatistics,
)


class AnalyticsService:
    def __init__(self):
        self.statistics_analyzer = DescriptiveStatistics()

    def generate_statistics(self, df):

        statistics = self.statistics_analyzer.generate(df)

        return statistics
