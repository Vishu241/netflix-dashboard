from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

class NetflixAnalytics:
    """
    Object-oriented API for Netflix titles EDA and plotting.

    Parameters
    ----------
    data_path : str | os.PathLike
        Path to `netflix_titles.csv` or `.xlsx`.
    output_dir : str | os.PathLike, optional
        Where to save figures/reports. Defaults to data file's folder / 'outputs'.
    """

    def __init__(self, data_path: os.PathLike | str, output_dir: os.PathLike | str | None = None):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")
        self.output_dir = Path(output_dir) if output_dir else self.data_path.parent / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df: Optional[pd.DataFrame] = None
        self._prepared: bool = False
        self._content_by_year: Optional[pd.DataFrame] = None
        self._type_counts: Optional[pd.Series] = None
        self._rating_counts: Optional[pd.Series] = None
        self._release_counts: Optional[pd.Series] = None
        self._country_counts: Optional[pd.Series] = None
        self._movie_df: Optional[pd.DataFrame] = None

    # ---------- Public API ----------

    def load(self) -> "NetflixAnalytics":
        """Load CSV/XLSX with auto-detection."""
        suffix = self.data_path.suffix.lower()
        if suffix == ".csv":
            self.df = pd.read_csv(self.data_path)
        elif suffix in (".xlsx", ".xls"):
            try:
                self.df = pd.read_excel(self.data_path)
            except ImportError as e:
                raise ImportError("Reading Excel requires `openpyxl` (pip install openpyxl).") from e
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        logging.info(f"Loaded {len(self.df):,} rows from {self.data_path.name}")
        return self

    def prepare(self) -> "NetflixAnalytics":
        """Minimal cleaning + derived cols used across charts."""
        self._require_df()
        df = self.df.copy()

        # Standard columns that many public datasets have
        needed = [c for c in ['type', 'release_year', 'rating', 'country', 'duration', 'listed_in', 'date_added'] if c in df.columns]
        df = df.dropna(subset=[c for c in needed if c != 'date_added'])  # allow missing date_added

        # Duration (minutes) for movies
        if 'duration' in df.columns:
            movie_mask = df['type'].eq('Movie')
            # Extract leading integer before ' min'
            df.loc[movie_mask, 'duration_int'] = (
                df.loc[movie_mask, 'duration']
                  .astype(str)
                  .str.extract(r'(\d+)', expand=False)
                  .astype('float')
            )

        # Parse date_added to monthly bucket if available
        if 'date_added' in df.columns:
            # Handle both "September 9, 2019" and pandas-friendly formats
            # Safe, modern datetime parsing
            df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce', utc=True)

            # Convert to naive (no tz) before period bucketing to avoid tz warning
            date_naive = df['date_added'].dt.tz_convert(None)
            df['month_added'] = date_naive.dt.to_period('M').astype(str)


        self.df = df
        self._prepared = True
        # Reset caches
        self._content_by_year = None
        self._type_counts = None
        self._rating_counts = None
        self._release_counts = None
        self._country_counts = None
        self._movie_df = None
        logging.info("Data prepared.")
        return self

    # ---------- Summary/metrics ----------

    def summary(self) -> pd.DataFrame:
        """Return a compact numeric summary for quick export."""
        self._require_prepared()
        s = {
            "n_rows": len(self.df),
            "n_movies": int((self.df['type'] == 'Movie').sum()) if 'type' in self.df else 0,
            "n_tvshows": int((self.df['type'] == 'TV Show').sum()) if 'type' in self.df else 0,
            "min_year": int(self.df['release_year'].min()) if 'release_year' in self.df else None,
            "max_year": int(self.df['release_year'].max()) if 'release_year' in self.df else None,
            "n_ratings": self.df['rating'].nunique() if 'rating' in self.df else 0,
            "n_countries": self._explode_countries().nunique() if 'country' in self.df else 0,
        }
        return pd.DataFrame([s])

    # ---------- Plotters (each saves a file and returns Axes) ----------

    def plot_type_bar(self):
        ax = self._new_ax((6, 4), "movies_vs_tvshows.png")
        tc = self._get_type_counts()
        ax.bar(tc.index, tc.values)
        ax.set_title("Movies vs TV Shows")
        ax.set_xlabel("Type"); ax.set_ylabel("Count")
        self._finalize(ax)
        return ax

    def plot_rating_pie(self):
        ax = self._new_ax((8, 6), "content_ratings_pie.png")
        rc = self._get_rating_counts()
        ax.pie(rc, labels=rc.index, autopct='%1.1f%%', startangle=90)
        ax.set_title("Percentage of Content Ratings"); ax.axis('equal')
        self._finalize(ax)
        return ax

    def plot_movie_duration_hist(self, bins: int = 30):
        ax = self._new_ax((8, 6), "movie_duration_histogram.png")
        md = self._get_movie_df()
        if md.empty or 'duration_int' not in md.columns or md['duration_int'].isna().all():
            ax.text(0.5, 0.5, "No movie duration data", ha="center", va="center")
        else:
            ax.hist(md['duration_int'].dropna().astype(float), bins=bins, edgecolor='black')
            ax.set_title("Distribution of Movie Duration")
            ax.set_xlabel("Duration (minutes)"); ax.set_ylabel("Number of Movies")
        self._finalize(ax)
        return ax

    def plot_release_scatter(self):
        ax = self._new_ax((10, 6), "release_year_scatter.png")
        rc = self._get_release_counts()
        ax.scatter(rc.index, rc.values, s=12)
        ax.set_title("Release Year vs Number of Shows")
        ax.set_xlabel("Release Year"); ax.set_ylabel("Number of Shows")
        self._finalize(ax)
        return ax

    def plot_top_countries(self, top_n: int = 10, split_multi: bool = True):
        ax = self._new_ax((8, 6), "top_countries.png")
        cc = self._get_country_counts(split_multi=split_multi).head(top_n)
        ax.barh(cc.index[::-1], cc.values[::-1])
        ax.set_title(f"Top {top_n} Countries by Number of Shows")
        ax.set_xlabel("Number of Shows"); ax.set_ylabel("Country")
        self._finalize(ax)
        return ax

    def plot_year_trends(self):
        ax = self._new_ax((8, 6), "releases_per_year.png")
        cby = self._get_content_by_year()
        ax.plot(cby.index, cby.get('Movie', pd.Series([0]*len(cby), index=cby.index)), label="Movies")
        ax.plot(cby.index, cby.get('TV Show', pd.Series([0]*len(cby), index=cby.index)), label="TV Shows")
        ax.set_title("Releases per Year"); ax.set_xlabel("Year"); ax.set_ylabel("Count")
        ax.legend()
        self._finalize(ax)
        return ax

    # ---- New extras ----

    def plot_top_genres(self, top_n: int = 12):
        """Split 'listed_in' and show top genres if available."""
        ax = self._new_ax((10, 6), "top_genres.png")
        if 'listed_in' not in self.df.columns:
            ax.text(0.5, 0.5, "No 'listed_in' (genres) column", ha="center", va="center")
        else:
            genres = (self.df['listed_in']
                        .astype(str)
                        .str.split(',')
                        .explode()
                        .str.strip()
                        .value_counts()
                        .head(top_n))
            ax.barh(genres.index[::-1], genres.values[::-1])
            ax.set_title(f"Top {top_n} Genres")
            ax.set_xlabel("Count"); ax.set_ylabel("Genre")
        self._finalize(ax)
        return ax

    def plot_rating_mix_by_type(self, top_n: int = 10):
        """Stacked bar: rating distribution within Types."""
        ax = self._new_ax((10, 6), "rating_mix_by_type.png")
        if not {'type', 'rating'}.issubset(self.df.columns):
            ax.text(0.5, 0.5, "Need 'type' and 'rating'", ha="center", va="center")
        else:
            mix = (self.df.groupby(['rating', 'type']).size()
                         .unstack(fill_value=0)
                         .sort_values('Movie', ascending=False)
                         .head(top_n))
            mix.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f"Rating Mix by Type (Top {top_n} Ratings)")
            ax.set_xlabel("Rating"); ax.set_ylabel("Count")
        self._finalize(ax)
        return ax

    def plot_monthly_added(self):
        """Monthly additions trend if 'date_added' exists."""
        ax = self._new_ax((10, 6), "monthly_added.png")
        if 'month_added' not in self.df.columns:
            ax.text(0.5, 0.5, "No 'date_added' column", ha="center", va="center")
        else:
            monthly = self.df.groupby('month_added').size().sort_index()
            ax.plot(pd.to_datetime(monthly.index), monthly.values)
            ax.set_title("Titles Added per Month")
            ax.set_xlabel("Month"); ax.set_ylabel("Count")
        self._finalize(ax)
        return ax

    # ---------- Dashboard ----------

    def save_dashboard(self, filename_base: str = "netflix_dashboard"):
        """Create a 3x3 dashboard (6 core + 3 optional if available)."""
        self._require_prepared()
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle("Netflix Titles Dashboard", fontsize=18)

        # row 1
        self._plot_on_axes(self.plot_type_bar, axes[0, 0])
        self._plot_on_axes(self.plot_rating_pie, axes[0, 1])
        self._plot_on_axes(self.plot_movie_duration_hist, axes[0, 2])

        # row 2
        self._plot_on_axes(self.plot_release_scatter, axes[1, 0])
        self._plot_on_axes(self.plot_top_countries, axes[1, 1])
        self._plot_on_axes(self.plot_year_trends, axes[1, 2])

        # row 3 (extras; safe if columns missing)
        self._plot_on_axes(self.plot_top_genres, axes[2, 0])
        self._plot_on_axes(self.plot_rating_mix_by_type, axes[2, 1])
        self._plot_on_axes(self.plot_monthly_added, axes[2, 2])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        png = self.output_dir / f"{filename_base}.png"
        pdf = self.output_dir / f"{filename_base}.pdf"
        fig.savefig(png, dpi=200)
        fig.savefig(pdf)
        plt.close(fig)
        logging.info(f"Saved dashboard -> {png.name}, {pdf.name}")

    # ---------- Exports ----------

    def export_summary(self, csv_name: str = "summary.csv", json_name: str = "summary.json"):
        """Save summary table as CSV + JSON in output_dir."""
        s = self.summary()
        s.to_csv(self.output_dir / csv_name, index=False)
        s.to_json(self.output_dir / json_name, orient="records", indent=2)
        logging.info(f"Saved summary -> {csv_name}, {json_name}")

    # ---------- Internals ----------

    def _new_ax(self, figsize: Tuple[int, int], fname: str):
        self._require_prepared()
        fig, ax = plt.subplots(figsize=figsize)
        ax._file_to_save = self.output_dir / fname  # attach for _finalize
        return ax

    def _finalize(self, ax):
        plt.tight_layout()
        fig = ax.get_figure()
        fig.savefig(ax._file_to_save)  # type: ignore[attr-defined]
        plt.close(fig)
        logging.info(f"Saved figure -> {ax._file_to_save.name}")

    def _plot_on_axes(self, plotter, ax):
        """Run a plotter but draw onto an existing Axes (for dashboard)."""
        # Monkey-patch _new_ax to use the provided axes temporarily.
        orig_new_ax = self._new_ax

        def _new_ax_override(figsize, fname):
            # reuse provided ax; store target path for consistency
            ax._file_to_save = self.output_dir / fname  # type: ignore[attr-defined]
            return ax

        self._new_ax = _new_ax_override  # type: ignore[method-assign]
        try:
            plotter()  # will draw on current ax
        finally:
            self._new_ax = orig_new_ax  # restore

    def _get_type_counts(self) -> pd.Series:
        if self._type_counts is None:
            self._type_counts = self.df['type'].value_counts()
        return self._type_counts

    def _get_rating_counts(self) -> pd.Series:
        if self._rating_counts is None:
            self._rating_counts = self.df['rating'].value_counts()
        return self._rating_counts

    def _get_release_counts(self) -> pd.Series:
        if self._release_counts is None:
            self._release_counts = self.df['release_year'].value_counts().sort_index()
        return self._release_counts

    def _get_movie_df(self) -> pd.DataFrame:
        if self._movie_df is None:
            self._movie_df = self.df[self.df['type'] == 'Movie'].copy() if 'type' in self.df.columns else pd.DataFrame()
        return self._movie_df

    def _explode_countries(self) -> pd.Series:
        # Split multi-country strings into separate rows
        if 'country' not in self.df.columns:
            return pd.Series(dtype=str)
        return (self.df['country'].astype(str).str.split(',')
                    .explode().str.strip())

    def _get_country_counts(self, split_multi: bool = True) -> pd.Series:
        if split_multi:
            return self._explode_countries().value_counts()
        if self._country_counts is None:
            self._country_counts = self.df['country'].value_counts()
        return self._country_counts

    def _get_content_by_year(self) -> pd.DataFrame:
        if self._content_by_year is None:
            self._content_by_year = (self.df.groupby(['release_year', 'type'])
                                         .size().unstack().fillna(0))
        return self._content_by_year

    def _require_df(self):
        if self.df is None:
            raise RuntimeError("Call .load() first.")

    def _require_prepared(self):
        self._require_df()
        if not self._prepared:
            raise RuntimeError("Call .prepare() after .load().")
