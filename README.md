The codebase provides a sensor-agnostic framework to calculate local spectral complexity from multi-temporal satellite imagery. By establishing a common geographic grid formatted to HDF-EOS5 standards, the codebase processes Landsat surface reflectance data and hyperspectral data from the Tanager-1 satellite. It evaluates spectral complexity by extracting local endmembers and calculating the volume of their resulting parallelotope via the Gram matrix determinant.

## Files:

gee-landsat-to-hdf5-stacker.py: Queries and combines Landsat 8/9 data from Google Earth Engine. It reprojects scenes to a common UTM master grid, applies radiometric scaling locally, and structures the output into an HDF-EOS5 compliant spatial-temporal stack.

Tanager_HDF5_stacker.py: Projects local Tanager orthorectified datasets into the identical HDF-EOS5 master grid established by the Landsat stacker, enabling spatially similar per-pixel analysis across both sensor platforms.

SpecComplex.py: The core mathematical library. Contains sensor-specific spatial masking algorithms utilizing QA and uncertainty bands, endmember extraction via maximum distance projection, and the formulation of parallelotope volumes.

spectral_complexity_calculations.py: Iterates over the temporally stacked HDF5 files to calculate the sliding-window spectral complexity and appending the resulting arrays directly into the source HDF5 files.

h5_viewer.py: A local validation script utilizing matplotlib to render the ortho-visual layers alongside the generated sliding volume maps.

## References

[1] Earth Resources Observation and Science (EROS) Center, “Landsat 8-9 Operational Land Imager / Thermal Infrared Sensor Level-2, Collection 2 [dataset].” Nov. 09, 2022. doi: 10.5066/P9OGBGM6.

[2] N. S. Chiaratti and D. W. Messinger, "Sub-Pixel Activity Monitoring in Landsat via Temporal Spectral Complexity," To be published 2026.

[3] N. Gorelick, M. Hancher, M. Dixon, S. Ilyushchenko, D. Thau, and R. Moore, “Google Earth Engine: Planetary-scale geospatial analysis for everyone,” Remote Sensing of Environment, vol. 202, pp. 18–27, Dec. 2017, doi: 10.1016/j.rse.2017.06.031.

[4] D. W. Messinger, Amanda K. Ziemann, Ariel Schlamm, William Basener, "Metrics of spectral image complexity with application to large area search," Opt. Eng. 51(3) 036201 (29 March 2012) https://doi-org.ezproxy.rit.edu/10.1117/1.OE.51.3.036201

[5] Planet Labs PBC, “Planet Application Program Interface: In Space for Life on Earth.” 2025. [Online]. Available: https://api.planet.com

[6] J. R. Schott, K. Lee, R. Raqueno, G. Hoffmann, and G. Healey, “A Subpixel Target Detection Technique Based on the Invariance Approach.,” in AVIRIS Airborne Geoscience workshop Proceedings, 2003. [Online]. Available: https://popo.jpl.nasa.gov/pub/docs/workshops/03_docs/Schott_AVIRIS_2003_web.pdf

[7] Y. Zha, J. Gao, and S. Ni, “Use of normalized difference built-up index in automatically mapping urban areas from TM imagery,” International Journal of Remote Sensing - INT J REMOTE SENS, vol. 24, pp. 583–594, Feb. 2003, doi: 10.1080/01431160304987.



