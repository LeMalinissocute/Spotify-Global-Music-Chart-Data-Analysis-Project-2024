from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, desc, count, avg, max as _max, min as _min, stddev, abs as _abs, lit, split, explode, trim
from pyspark.sql.types import DoubleType
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

spark = SparkSession.builder.appName("SpotifyAnalysis").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("Spotify Data Analysis")
print("=" * 60)

df = spark.read.parquet("hdfs://localhost:9000/user/spotify/spotify_clean.parquet")
print(f"Total rows: {df.count():,}")
print(f"Unique tracks: {df.select('track_name').distinct().count():,}")
print(f"Unique artists: {df.select('artist_name').distinct().count():,}")

# ============================================
# 1. Top 10 歌曲播放量统计
# ============================================
print("\n1. Top 10 歌曲播放量统计...")

top_tracks = df.groupBy("track_name", "artist_name").agg(_sum("streams").alias("total_streams")).orderBy(desc("total_streams")).limit(10)
top_tracks_pd = top_tracks.toPandas()
top_tracks_pd.to_csv("result_csv/top_tracks.csv", index=False, encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(top_tracks_pd)), top_tracks_pd['total_streams']/1e6, color='#2E86AB')
ax.set_yticks(range(len(top_tracks_pd)))
ax.set_yticklabels([f"{row['track_name']}" for _, row in top_tracks_pd.iterrows()], fontsize=10)
ax.set_xlabel("Total Streams (Millions)", fontsize=12)
ax.set_title("Top 10 Most Streamed Songs on Spotify 2024", fontsize=14, fontweight='bold')
ax.invert_yaxis()
for i, bar in enumerate(bars):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.0f}M", va='center', fontsize=9)
fig.savefig("result_pic/top_tracks.png", dpi=300, bbox_inches='tight')
print("  ✅ top_tracks.csv / top_tracks.png")

# ============================================
# 2. Top 10 歌手播放量统计
# ============================================
print("\n2. Top 10 歌手播放量统计...")

artists_df = df.withColumn("artist_name", explode(split(col("artist_name"), ","))) \
    .withColumn("artist_name", trim(col("artist_name"))) \
    .filter(col("artist_name") != "")

top_artists = artists_df.groupBy("artist_name").agg(_sum("streams").alias("total_streams")).orderBy(desc("total_streams")).limit(10)
top_artists_pd = top_artists.toPandas()
top_artists_pd.to_csv("result_csv/top_artists.csv", index=False, encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(top_artists_pd)), top_artists_pd['total_streams']/1e6, color='#A23B72')
ax.set_yticks(range(len(top_artists_pd)))
ax.set_yticklabels([f"{row['artist_name']}" for _, row in top_artists_pd.iterrows()], fontsize=10)
ax.set_xlabel("Total Streams (Millions)", fontsize=12)
ax.set_title("Top 10 Most Streamed Artists on Spotify 2024", fontsize=14, fontweight='bold')
ax.invert_yaxis()
for i, bar in enumerate(bars):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.0f}M", va='center', fontsize=9)
fig.savefig("result_pic/top_artists.png", dpi=300, bbox_inches='tight')
print("  ✅ top_artists.csv / top_artists.png")

# ============================================
# 3. 月度歌曲热度分析
# ============================================
print("\n3. 月度歌曲热度分析...")

monthly = df.withColumn("month", col("chart_date").substr(1, 7)).groupBy("month").agg(_sum("streams").alias("total_streams"), avg("streams").alias("avg_streams")).orderBy("month")
monthly_pd = monthly.toPandas()
monthly_pd.to_csv("result_csv/monthly_streams.csv", index=False, encoding='utf-8-sig')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
ax1.plot(monthly_pd['month'], monthly_pd['total_streams']/1e9, 'o-', color='#2E86AB', linewidth=2, markersize=8)
ax1.set_xlabel("Month", fontsize=12)
ax1.set_ylabel("Total Streams (Billions)", fontsize=12)
ax1.set_title("Monthly Total Streams Trend", fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

ax2.bar(monthly_pd['month'], monthly_pd['avg_streams']/1e6, color='#E84855', alpha=0.7)
ax2.set_xlabel("Month", fontsize=12)
ax2.set_ylabel("Avg Streams (Millions)", fontsize=12)
ax2.set_title("Monthly Average Streams Trend", fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)
fig.savefig("result_pic/monthly_trends.png", dpi=300, bbox_inches='tight')
print("  ✅ monthly_streams.csv / monthly_trends.png")

# ============================================
# 4. 歌曲在榜周数分析
# ============================================
print("\n4. 歌曲在榜周数分析...")

weeks = df.groupBy("weeks_on_chart").agg(count("*").alias("song_count")).orderBy("weeks_on_chart")
weeks_pd = weeks.toPandas()
weeks_pd.to_csv("result_csv/weeks_distribution.csv", index=False, encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(14, 6))
weeks_filtered = weeks_pd[weeks_pd['weeks_on_chart'] <= 52]
ax.bar(weeks_filtered['weeks_on_chart'], weeks_filtered['song_count'], color='#4A7C59')
ax.set_xlabel("Weeks on Chart", fontsize=12)
ax.set_ylabel("Number of Songs", fontsize=12)
ax.set_title("Distribution of Songs by Weeks on Chart", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
fig.savefig("result_pic/weeks_distribution.png", dpi=300, bbox_inches='tight')
print("  ✅ weeks_distribution.csv / weeks_distribution.png")

# ============================================
# 5. 歌曲排名趋势分析 previous_rank - rank
# ============================================
print("\n5. 歌曲排名趋势分析...")

df_with_rank_change = df.withColumn(
    'rank_change',
    col('previous_rank') - col('rank')
)

rank_trend = df_with_rank_change.groupBy('track_name', 'artist_name') \
    .agg(
        avg('rank_change').alias('avg_rank_change'),
        _sum('streams').alias('total_streams'),
        count('*').alias('weeks_appeared')
    ) \
    .filter(col('weeks_appeared') >= 4) \
    .orderBy(desc('avg_rank_change')) \
    .limit(10)

rank_trend_pd = rank_trend.toPandas()
rank_trend_pd.to_csv("result_csv/rank_trend.csv", index=False, encoding='utf-8-sig')

if len(rank_trend_pd) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in rank_trend_pd['avg_rank_change']]
    bars = ax.barh(range(len(rank_trend_pd)), rank_trend_pd['avg_rank_change'], color=colors)
    ax.set_yticks(range(len(rank_trend_pd)))
    ax.set_yticklabels([f"{row['track_name']}" for _, row in rank_trend_pd.iterrows()], fontsize=10)
    ax.set_xlabel("Average Rank Change (positive = improving)", fontsize=12)
    ax.set_title("Songs with Best Rank Improvement Trend", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.3 if bar.get_width() > 0 else bar.get_width() - 1,
                bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.1f}", va='center', ha='left' if bar.get_width() > 0 else 'right', fontsize=9)

    fig.savefig("result_pic/rank_trend.png", dpi=300, bbox_inches='tight')
    print("  ✅ rank_trend.csv / rank_trend.png")

# ============================================
# 6. 黑马歌曲分析 rank <= 10
# ============================================
print("\n6. 黑马歌曲分析...")

dark_horse = df.filter(col("rank") <= 10).groupBy("track_name", "artist_name").agg(_min("weeks_on_chart").alias("min_weeks_on_chart"), count("*").alias("appearances")).orderBy("min_weeks_on_chart").limit(20)
dark_horse_pd = dark_horse.toPandas()
dark_horse_pd.to_csv("result_csv/dark_horse_songs.csv", index=False, encoding='utf-8-sig')

if len(dark_horse_pd) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(dark_horse_pd)), dark_horse_pd['min_weeks_on_chart'], color='#E67E22')
    ax.set_yticks(range(len(dark_horse_pd)))
    ax.set_yticklabels([f"{row['track_name']}" for _, row in dark_horse_pd.iterrows()], fontsize=9)
    ax.set_xlabel("Minimum Weeks on Chart", fontsize=12)
    ax.set_title("Dark Horse Songs (Rank <= 10, Minimum Weeks on Chart)", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.0f}", va='center', fontsize=9)
    fig.savefig("result_pic/dark_horse_songs.png", dpi=300, bbox_inches='tight')
    print("  ✅ dark_horse_songs.csv / dark_horse_songs.png")

# ============================================
# 7. 稳定性分析 标准差
# ============================================
print("\n7. 稳定性分析...")

stability = df.groupBy("track_name", "artist_name").agg(stddev("rank").alias("rank_stddev"), count("*").alias("weeks_on_chart"), avg("rank").alias("avg_rank")).filter(col("weeks_on_chart") >= 10).orderBy("rank_stddev").limit(20)
stability_pd = stability.toPandas()
stability_pd.to_csv("result_csv/stability_analysis.csv", index=False, encoding='utf-8-sig')
print("  ✅ stability_analysis.csv")

# ============================================
# 8. 唱片公司分析
# ============================================
print("\n8. 唱片公司分析...")

label_stats = df.groupBy("source").agg(_sum("streams").alias("total_streams"), count("*").alias("appearances")).orderBy(desc("total_streams")).limit(20)
label_stats_pd = label_stats.toPandas()
total_streams_all = label_stats_pd['total_streams'].sum()
label_stats_pd['percentage'] = (label_stats_pd['total_streams'] / total_streams_all * 100).round(2)
label_stats_pd.to_csv("result_csv/label_analysis.csv", index=False, encoding='utf-8-sig')

if len(label_stats_pd) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    top_labels = label_stats_pd.head(10)
    ax1.barh(top_labels['source'], top_labels['total_streams']/1e9, color='#3498DB')
    ax1.set_xlabel("Total Streams (Billions)", fontsize=12)
    ax1.set_title("Top 10 Labels by Total Streams", fontsize=14, fontweight='bold')

    top_labels_pie = label_stats_pd.head(8)
    ax2.pie(top_labels_pie['total_streams'], labels=top_labels_pie['source'], autopct='%1.1f%%', startangle=90)
    ax2.set_title("Market Share by Label (Top 8)", fontsize=14, fontweight='bold')
    fig.savefig("result_pic/label_analysis.png", dpi=300, bbox_inches='tight')
    print("  ✅ label_analysis.csv / label_analysis.png")

print("\n" + "=" * 60)
print("Analysis completed successfully!")
print("Results saved to result_csv/ and result_pic/")
print("=" * 60)

spark.stop()