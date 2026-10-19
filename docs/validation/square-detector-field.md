# Square cuvette: detector field correction after v0.3.1

The square mode in v0.3.1 clipped the incident ray field to the outer side L.
A square rotated by theta has horizontal width
`L * (abs(cos(theta)) + abs(sin(theta)))`; its maximum is `sqrt(2)*L`.
The earlier central-chord test did not detect the missing peripheral rays.

For a 50 mm outer side and 46 mm inner side, the inner corner can project as
far as 32.53 mm from the axis, beyond the old field boundary at 25 mm. A
finite target patch near (21, -21) was entirely absent at 30, 45 and 60 degrees
in the straight-ray reproduction (equal refractive indices).

The corrected square mode uses one constant detector span `sqrt(2)*L` at all
angles: 70.71 mm for L=50 mm. Using the maximum rather than a different span
at each angle preserves a single detector pitch and a consistent projector
calibration across frames. Cylindrical modes retain the outer diameter.

The target voxel grid still spans the original side, not the diagonal. Export
uses the detector span for horizontal physical scale and frame aspect ratio;
vertical sampling uses the finer of voxel-grid and projector pixel pitches.
Preflight uses the same geometry. The manifest records detector span, detector
spacing, PNG pixel pitch, voxel-grid span, Z pitch and actual raster layer count.
Readers use that layer count when available and retain the legacy estimate for
older manifests. The metadata schema and old frame-set files remain readable.

Voxel resolution, detector-bin count and PNG width are unchanged. Expanding the
field at a fixed pixel count increases physical detector/pixel pitch; it does
not create extra spatial resolution or expand the actual optical hardware's
field. A physical projector must be calibrated to the recorded span. If it can
illuminate only a narrower field, corner coverage cannot be repaired by software
alone. Refraction can also limit coverage independently of field truncation.

Regression checks:

- all four finite corner patches receive rays at every 5-degree view, with
  equal indices to isolate field coverage from refractive effects;
- every detector chord agrees with an independent slab-intersection reference
  across multiple square rotations, including the outer detector region;
- an offset corner model survives PNG export at 45 degrees, with expected
  detector position, frame aspect ratio, sampling pitch and manifest geometry;
- existing circular-vat, absorption, Python/C++ parity and export tests remain.

Run:

```bash
python -m pytest tests/test_physical_optics.py tests/test_physical_export.py
```
