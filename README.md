# IGC to Video Overlay

Video overlay generator for paragliding flights from IGC files.

## Description

This project generates video overlays displaying real-time flight data:
- **Time-series graphs**: Altitude, Vario (vertical speed), Ground Speed
- **Flight information**: Time, flight duration, distance traveled
- **Time window**: ±5 minutes of data around the current moment

## Prerequisites

```bash
pip install mediapy pixie-python numpy matplotlib aerofiles scipy pillow requests tqdm timezonefinder pytz
```

## Configuration

Edit the configuration section in `igc_to_overlay.py`:

```python
# TEST MODE
TEST_MODE = False  # True = Test image, False = Full video

# Parameters
TOTAL_FLIGHT_DIST = 78  # Total distance (km) with turnpoints
speed_acc = 16  # Video acceleration factor (16x)

# IGC Source (URL or local path)
file_url = None  # or "https://..."
file_path = r"C:\path\to\file.igc"

qualite_compression = 15  # Video quality (10-20, higher = better quality)

# Performance
USE_PARALLEL = True  # Enable multiprocessing for faster frame generation
NUM_WORKERS = cpu_count()  # Number of CPU cores to use
```

**Note**: Timezone is automatically detected from GPS coordinates - no manual adjustment needed!

## Usage

### Test Mode
Generates a test image to verify layout:

```python
TEST_MODE = True
```

Then run:
```bash
python igc_to_overlay.py
```

**Output**: `test_overlay.png` - Frame 30 of a simulated flight (60 points)

### Video Mode
Generates the complete video with overlay:

```python
TEST_MODE = False
file_path = r"C:\path\to\your_flight.igc"
```

Then run:
```bash
python igc_to_overlay.py
```

**Output**: `output/your_flight_overlay.mp4` - Full video at 24 fps

The output video is automatically named based on the input IGC filename and saved in the `output/` folder.

## Output Structure

```
Overlay/
├── igc_to_overlay.py
├── test_overlay.png          # Test mode output
├── output/                    # Video mode outputs
│   ├── flight1_overlay.mp4
│   ├── flight2_overlay.mp4
│   └── ...
└── ...
```

## Features

### Automatic Timezone Detection
- Detects timezone from GPS coordinates at flight start
- Automatically handles Daylight Saving Time (DST)
- No manual timezone configuration required

### Graphs
- **Altitude** (green): Displays GPS altitude in meters
- **Vario** (orange): Vertical speed in m/s
- **Speed** (blue): Ground speed in km/h

Each graph displays:
- Time window: -5min to +5min around current moment
- White vertical marker at "Now"
- Yellow dot on current value
- Y-axis scales with graduations
- Current value displayed above the graph

### Display Information
- **HOUR**: Local time (automatically detected from GPS coordinates with DST support)
- **FLIGHT TIME**: Flight duration
- **FLIGHT DIST**: Distance traveled (interpolated)

## Code Structure

### Main Functions

**IGC Processing**
- `read_igc()`: Read and parse IGC file
- `smooth_igc_output()`: Data smoothing (Hanning window)
- `reshape_array()`: Interpolation to 24 fps

**Graph Generation**
- `draw_time_series_graph()`: Draw complete time-series graph
- `add_time_series_graphs()`: Add all 3 graphs to image
- `compute_graph_parameters()`: Calculate min/max scales

**Image Generation**
- `gen_img_from_smoothed_list()`: Frame generator for video
- `generate_dummy_igc()`: Create test IGC file

### Processing Pipeline

```
IGC File → Read → Smooth → Reshape (24fps) → Compute Graph Config
                                                      ↓
                                            Generate Frames
                                                      ↓
                                     Test: Save Frame 30 | Video: Save All Frames
```

## IGC Format

The project uses the standard IGC format for flight data:
- **B records**: GPS points (lat, lon, altitude, timestamp)
- Frequency: Typically 1 point/second
- Data is interpolated to 24 fps for video

## Resolution

- **Video**: 1920x1080 (Full HD)
- **Framerate**: 24 fps
- **Acceleration**: 16x (configurable)

## Examples


### Performance
Processing can take several minutes for a long flight:
- IGC reading: < 1s
- Smoothing: ~1s
- Frame generation: ~10s per minute of video
- Video encoding: Depends on ffmpeg


## License

Personal project for paragliding flight analysis.

## Author

Cédric Gerber, inspired by Will's code  https://www.youtube.com/@williglide5449
