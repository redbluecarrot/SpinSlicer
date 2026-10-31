"""Frames land 1:1 on the projector screen; the cone of rays is easy to set; the game works."""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from projector_canvas import CanvasGeometry, compose, cone_smear_mm, throw_distance_mm  # noqa: E402


def test_frame_is_pasted_one_to_one_in_the_middle() -> None:
    frame = np.full((100, 720), 200, np.uint8)
    canvas = compose(frame, CanvasGeometry(1920, 1080))
    assert canvas.shape == (1080, 1920)
    rows, cols = np.nonzero(canvas)
    assert (cols.min(), cols.max() + 1) == (600, 1320)
    assert (rows.min(), rows.max() + 1) == (490, 590)
    assert canvas.sum() == frame.sum()          # no resampling, no stretching


def test_offset_moves_and_crops() -> None:
    frame = np.full((10, 20), 255, np.uint8)
    g = CanvasGeometry(40, 20, offset_x=25)
    assert not g.fits(20, 10)
    canvas = compose(frame, g)
    assert canvas[5:15, 35:].all() and not canvas[:, :35].any()
    assert canvas.sum() < frame.sum()


def test_throw_ratio_to_distance() -> None:
    # Vial 50 mm over 1000 px of a 1920 px screen → picture 96 mm wide; ratio 1.5 → 144 mm.
    assert np.isclose(throw_distance_mm(1.5, 50, 1000, 1920), 144.0)
    assert cone_smear_mm(0, 10, 20) == 0.0
    assert cone_smear_mm(200, 10, 20) > cone_smear_mm(400, 10, 20)


def test_export_writes_the_projector_size(tmp_path) -> None:
    from PyQt6.QtCore import QCoreApplication

    _app = QCoreApplication.instance() or QCoreApplication([])
    from workers import VideoExportWorker, _ffmpeg_executable

    if _ffmpeg_executable() is None:
        return
    frames = [np.full((40, 64), 120, np.uint8) for _ in range(4)]
    out = tmp_path / "v.mp4"
    worker = VideoExportWorker(frames, 24.0, str(out), canvas=CanvasGeometry(320, 180))
    errors: list[str] = []
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors
    import cv2
    cap = cv2.VideoCapture(str(out))
    ok, image = cap.read()
    assert ok and image.shape[:2] == (180, 320)


def test_game_cures_the_part_silhouette() -> None:
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    import trimesh

    from pixel_game import PixelGame, silhouette_from_mesh

    box = trimesh.creation.box(extents=(20, 10, 10))
    grid = silhouette_from_mesh(box)
    assert grid.sum() > 20
    game = PixelGame()
    game.set_shape(grid)
    rows, cols = np.nonzero(grid)
    game.simulate_hit(int(rows.max()), int(cols[rows.argmax()]))
    assert 0 < game.cured_fraction() < 1


def test_brightness_scales_light_not_gray_code() -> None:
    from video_tab import brightness_lut
    lut = brightness_lut(0.5, 2.2)
    light = (lut.astype(float) / 255) ** 2.2
    full = (np.arange(256) / 255) ** 2.2
    assert lut[0] == 0 and lut[255] == round(255 * 0.5 ** (1 / 2.2))
    np.testing.assert_allclose(light[64:], 0.5 * full[64:], rtol=0.03)   # half the light
    assert (brightness_lut(1.0, 2.2) == np.arange(256)).all()


def test_projector_frame_follows_the_clock_and_stops_after_the_dose() -> None:
    """The shown frame is ⌊t·fps⌋ (no drift from timer rounding); the window closes itself
    when the planned exposure time is over and reports it as completed."""
    import time

    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import QApplication

    from video_tab import ProjectorOutputWindow
    app = QApplication.instance() or QApplication([])
    frames = [np.full((8, 16), i, np.uint8) for i in range(24)]
    window = ProjectorOutputWindow(frames, CanvasGeometry(64, 32), 24.0,
                                   QGuiApplication.primaryScreen(), duration_s=0.6)
    done: list[bool] = []
    window.finished.connect(done.append)
    start = time.monotonic()
    seen = []
    while not done and time.monotonic() - start < 3:
        app.processEvents()
        if window._shown >= 0:
            seen.append((window.elapsed_s(), window._shown))
        time.sleep(0.005)
    assert done == [True]
    assert 0.55 < time.monotonic() - start < 1.5
    for t, shown in seen[5:]:
        assert abs(shown - int(t * 24) % 24) <= 1


def test_layers_pieces_stay_put_until_an_arrow_is_pressed() -> None:
    """New pieces spawn in the middle and do not slide towards a resting mouse cursor."""
    from PyQt6.QtCore import Qt

    from pixel_game import Layers
    game = Layers()
    game.state = "play"
    start = game.px
    for x in (0.0, 5.0, 159.0):
        game.mouse_x(x)
    assert game.px == start
    for _ in range(5):                     # drop pieces; every new one starts at the same column
        game.key_press(Qt.Key.Key_Space)
        game.flash = 0
        if game.over:
            break
        assert game.px == game.BW // 2 - 2
    game.key_press(Qt.Key.Key_Left)
    assert game.px == game.BW // 2 - 3
