To regenerate the postal comparison maps from `Visualization/`, you need:

1. `predictions_all.csv`
2. `data/us_zip_codes_geo.json`
3. Neighborhood GeoJSON files in `airbnb_geojson/`:
   - `ny_nei.geojson`
   - `la_nei.geojson`
   - `sf_nei.geojson`
   - `sd_nei.geojson`

Run:

```bash
./run_maps.sh
```

Output maps are written to `Visualization/maps/`.

Pipeline:

1. Read `predictions_all.csv` (model scores for Airbnb + hotel rows).
2. Group hotel rows by ZIP and Airbnb rows by neighborhood.
3. Convert Airbnb neighborhood scores into ZIP scores using neighborhood/ZIP overlap.
4. Compare Airbnb vs hotel per ZIP:
   - positive diff -> pink (Airbnb stronger)
   - negative diff -> blue (hotel stronger)
   - missing data -> gray
