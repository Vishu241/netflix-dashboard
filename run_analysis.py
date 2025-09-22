from netflix_api import NetflixAnalytics

ana = (
    NetflixAnalytics(r"C:\Users\HP\Desktop\VISHAL THESIIS WORK\python\matplot\netflix_titles.csv")
    .load()
    .prepare()
)

# Individual figures (saved in /outputs)
ana.plot_type_bar()
ana.plot_rating_pie()
ana.plot_movie_duration_hist()
ana.plot_release_scatter()
ana.plot_top_countries(top_n=10, split_multi=True)
ana.plot_year_trends()

# Extras (auto-skip nicely if columns missing)
ana.plot_top_genres()
ana.plot_rating_mix_by_type()
ana.plot_monthly_added()

# Dashboard + summary
ana.save_dashboard()       # saves outputs/netflix_dashboard.png + .pdf
ana.export_summary()       # saves outputs/summary.csv + .json
print("Done. Check the 'outputs' folder.")
