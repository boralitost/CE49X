"""
CE49X - Wave Energy Farm Interference Animation
Explains why tighter WEC spacing boosts power via constructive wave interference.
Run with: manim -qh --format=mp4 wave_interference_animation.py WaveInterferenceScene
"""
from manim import *
import numpy as np

BG_COLOR = BLACK
OCEAN_BLUE = "#1a5276"
WAVE_CREST = "#5dade2"
WEC_COLOR = "#f39c12"
POWER_COLOR = "#e74c3c"
POWER_LOW = "#f1c40f"
CONSTRUCTIVE_COLOR = "#00ff88"


class WaveInterferenceScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        self.show_title()
        self.show_single_wec()
        self.show_tight_layout()
        self.show_spread_layout()
        self.show_comparison()

    # ── Act 1: Title ──────────────────────────────────────

    def show_title(self):
        title = Text(
            "Wave Energy: Why Spacing Matters",
            font_size=44,
            color=WHITE,
        )
        subtitle = Text(
            "Perth, Western Australia — Southern Ocean Swells",
            font_size=24,
            color=GRAY_B,
        )
        subtitle.next_to(title, DOWN, buff=0.4)

        self.play(Write(title), run_time=1)
        self.play(FadeIn(subtitle))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle))

    # ── Act 2: Incoming Swell + Single WEC ────────────────

    def show_single_wec(self):
        header = Text("How a Wave Energy Converter Works", font_size=34, color=WAVE_CREST)
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        # Wave phase tracker
        wave_phase = ValueTracker(0)

        # Ocean background
        ocean_rect = Rectangle(
            width=12, height=5, color=OCEAN_BLUE,
            fill_opacity=0.15, stroke_width=0,
        ).move_to(DOWN * 0.3)
        self.play(FadeIn(ocean_rect), run_time=0.3)

        # Incoming wavefronts: vertical bands that sweep left→right
        wavelength = 2.5  # screen units
        n_bands = 8
        wave_bands = VGroup()
        for i in range(n_bands):
            band = Line(
                UP * 2.5, DOWN * 2.5,
                stroke_width=4,
                stroke_opacity=0.6,
                color=WAVE_CREST,
            )
            wave_bands.add(band)

        def update_wave_bands(group):
            phase = wave_phase.get_value()
            for i, band in enumerate(group):
                x = -7 + (i * wavelength) + (phase % wavelength)
                band.move_to([x, -0.3, 0])
                # Fade bands near edges
                if x < -5.5 or x > 5.5:
                    band.set_stroke(opacity=0.1)
                else:
                    band.set_stroke(opacity=0.5)

        wave_bands.add_updater(update_wave_bands)
        self.add(wave_bands)

        # Direction arrow
        dir_arrow = Arrow(
            LEFT * 4.5 + DOWN * 2.5, RIGHT * -2.5 + DOWN * 2.5,
            color=WAVE_CREST, stroke_width=2, buff=0,
            max_tip_length_to_length_ratio=0.15,
        )
        dir_label = Text("Incoming swell", font_size=18, color=WAVE_CREST)
        dir_label.next_to(dir_arrow, DOWN, buff=0.15)
        self.play(GrowArrow(dir_arrow), FadeIn(dir_label), run_time=0.5)

        # Single WEC
        wec = Dot(ORIGIN + DOWN * 0.3, radius=0.15, color=WEC_COLOR, fill_opacity=1)
        wec_label = Text("WEC", font_size=16, color=WEC_COLOR)
        wec_label.next_to(wec, UP, buff=0.2)
        self.play(FadeIn(wec), FadeIn(wec_label), run_time=0.5)

        # Animate wave sweep and WEC pulsing
        # Start wave motion
        self.play(wave_phase.animate.set_value(2.5), run_time=2, rate_func=linear)

        # Spawn scattered ripples from WEC
        label = Text(
            "Each WEC absorbs wave energy AND radiates new waves",
            font_size=22,
            color=YELLOW,
        )
        label.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(label), run_time=0.5)

        # Ripple bursts while wave continues
        for burst in range(3):
            ripple = Circle(
                radius=0.2, color=CONSTRUCTIVE_COLOR,
                stroke_width=2.5, fill_opacity=0,
            ).move_to(wec.get_center())
            self.play(
                wave_phase.animate(rate_func=linear).increment_value(1.2),
                ripple.animate.scale(8).set_stroke(opacity=0),
                run_time=1.2,
            )
            self.remove(ripple)

        wave_bands.remove_updater(update_wave_bands)
        self.wait(0.5)

        # Cleanup
        all_act2 = VGroup(
            header, ocean_rect, wave_bands, dir_arrow, dir_label,
            wec, wec_label, label,
        )
        self.play(FadeOut(all_act2), run_time=0.6)

    # ── Act 3: Tight Layout (~70m spacing) ────────────────

    def show_tight_layout(self):
        header = Text("Tight Spacing (~70 m)", font_size=36, color=WAVE_CREST)
        header.to_edge(UP, buff=0.4)
        underline = Line(
            header.get_left() + DOWN * 0.15,
            header.get_right() + DOWN * 0.15,
            color=WAVE_CREST, stroke_width=2,
        )
        self.play(Write(header), Create(underline), run_time=0.6)

        # Ocean background
        ocean = Rectangle(
            width=12, height=4.5, color=OCEAN_BLUE,
            fill_opacity=0.12, stroke_width=0,
        ).move_to(DOWN * 0.1)
        self.add(ocean)

        # 3 WECs close together
        spacing = 1.5  # tight
        wec_positions = [LEFT * spacing, ORIGIN, RIGHT * spacing]
        wecs = VGroup()
        wec_labels = VGroup()
        for i, pos in enumerate(wec_positions):
            p = pos + DOWN * 0.1
            dot = Dot(p, radius=0.15, color=WEC_COLOR, fill_opacity=1)
            lbl = Text(f"WEC {i+1}", font_size=14, color=WEC_COLOR)
            lbl.next_to(dot, UP, buff=0.2)
            wecs.add(dot)
            wec_labels.add(lbl)

        self.play(
            LaggedStart(*[FadeIn(w) for w in wecs], lag_ratio=0.15),
            LaggedStart(*[FadeIn(l) for l in wec_labels], lag_ratio=0.15),
            run_time=0.6,
        )

        # Spacing indicator
        brace = BraceBetweenPoints(
            wecs[0].get_center() + DOWN * 0.4,
            wecs[1].get_center() + DOWN * 0.4,
            direction=DOWN, color=GRAY_B,
        )
        brace_lbl = Text("~70 m", font_size=16, color=GRAY_B)
        brace_lbl.next_to(brace, DOWN, buff=0.1)
        self.play(FadeIn(brace), FadeIn(brace_lbl), run_time=0.4)

        # Wave phase tracker
        wave_phase = ValueTracker(0)
        wavelength = 2.5

        # Wavefronts
        wave_bands = VGroup()
        for i in range(8):
            band = Line(
                UP * 2.2 + DOWN * 0.1, DOWN * 2.2 + DOWN * 0.1,
                stroke_width=3.5, stroke_opacity=0.4, color=WAVE_CREST,
            )
            wave_bands.add(band)

        def update_waves(group):
            phase = wave_phase.get_value()
            for i, band in enumerate(group):
                x = -7 + (i * wavelength) + (phase % wavelength)
                band.move_to([x, -0.1, 0])
                band.set_stroke(opacity=0.35 if -5.5 < x < 5.5 else 0.05)

        wave_bands.add_updater(update_waves)
        self.add(wave_bands)

        # Animate wave + ripples with constructive interference
        # Each WEC emits ripples; ripples reach neighbors while bright
        for cycle in range(3):
            ripples = []
            for wec in wecs:
                ripple = Circle(
                    radius=0.15, color=CONSTRUCTIVE_COLOR,
                    stroke_width=2, fill_opacity=0,
                ).move_to(wec.get_center())
                ripples.append(ripple)

            # Ripples expand — tight spacing means they overlap neighbors
            ripple_anims = []
            for r in ripples:
                ripple_anims.extend([
                    r.animate.scale(12).set_stroke(opacity=0),
                ])

            # Constructive interference flashes at WEC positions
            flashes = []
            for wec in wecs:
                flash = Dot(
                    wec.get_center(), radius=0.3,
                    color=CONSTRUCTIVE_COLOR, fill_opacity=0.6,
                )
                flashes.append(flash)

            self.play(
                wave_phase.animate(rate_func=linear).increment_value(2.0),
                *ripple_anims,
                *[Succession(
                    Wait(0.4),
                    FadeIn(f, run_time=0.3),
                    FadeOut(f, run_time=0.5),
                ) for f in flashes],
                run_time=1.5,
            )
            for r in ripples:
                self.remove(r)

        wave_bands.remove_updater(update_waves)

        # Constructive label
        boost_label = Text(
            "Scattered waves boost neighbors → higher total power",
            font_size=22, color=CONSTRUCTIVE_COLOR,
        )
        boost_label.to_edge(DOWN, buff=0.8)
        self.play(FadeIn(boost_label), run_time=0.5)

        # Power bars
        power_group = self._draw_power_bars(wecs, target_fill=0.88, label="~88%")
        self.wait(1)

        # Cleanup
        all_act3 = VGroup(
            header, underline, ocean, wecs, wec_labels,
            brace, brace_lbl, wave_bands, boost_label, power_group,
        )
        self.play(FadeOut(all_act3), run_time=0.6)

    # ── Act 4: Spread Layout (~110m spacing) ──────────────

    def show_spread_layout(self):
        header = Text("Wide Spacing (~110 m)", font_size=36, color=WAVE_CREST)
        header.to_edge(UP, buff=0.4)
        underline = Line(
            header.get_left() + DOWN * 0.15,
            header.get_right() + DOWN * 0.15,
            color=WAVE_CREST, stroke_width=2,
        )
        self.play(Write(header), Create(underline), run_time=0.6)

        # Ocean background
        ocean = Rectangle(
            width=12, height=4.5, color=OCEAN_BLUE,
            fill_opacity=0.12, stroke_width=0,
        ).move_to(DOWN * 0.1)
        self.add(ocean)

        # 3 WECs spread apart
        spacing = 3.5  # wide
        wec_positions = [LEFT * spacing, ORIGIN, RIGHT * spacing]
        wecs = VGroup()
        wec_labels = VGroup()
        for i, pos in enumerate(wec_positions):
            p = pos + DOWN * 0.1
            dot = Dot(p, radius=0.15, color=WEC_COLOR, fill_opacity=1)
            lbl = Text(f"WEC {i+1}", font_size=14, color=WEC_COLOR)
            lbl.next_to(dot, UP, buff=0.2)
            wecs.add(dot)
            wec_labels.add(lbl)

        self.play(
            LaggedStart(*[FadeIn(w) for w in wecs], lag_ratio=0.15),
            LaggedStart(*[FadeIn(l) for l in wec_labels], lag_ratio=0.15),
            run_time=0.6,
        )

        # Spacing indicator
        brace = BraceBetweenPoints(
            wecs[0].get_center() + DOWN * 0.4,
            wecs[1].get_center() + DOWN * 0.4,
            direction=DOWN, color=GRAY_B,
        )
        brace_lbl = Text("~110 m", font_size=16, color=GRAY_B)
        brace_lbl.next_to(brace, DOWN, buff=0.1)
        self.play(FadeIn(brace), FadeIn(brace_lbl), run_time=0.4)

        # Wave phase tracker
        wave_phase = ValueTracker(0)
        wavelength = 2.5

        wave_bands = VGroup()
        for i in range(8):
            band = Line(
                UP * 2.2 + DOWN * 0.1, DOWN * 2.2 + DOWN * 0.1,
                stroke_width=3.5, stroke_opacity=0.4, color=WAVE_CREST,
            )
            wave_bands.add(band)

        def update_waves(group):
            phase = wave_phase.get_value()
            for i, band in enumerate(group):
                x = -7 + (i * wavelength) + (phase % wavelength)
                band.move_to([x, -0.1, 0])
                band.set_stroke(opacity=0.35 if -5.5 < x < 5.5 else 0.05)

        wave_bands.add_updater(update_waves)
        self.add(wave_bands)

        # Ripples that fade before reaching neighbors (wide spacing)
        for cycle in range(3):
            ripples = []
            for wec in wecs:
                ripple = Circle(
                    radius=0.15, color=WAVE_CREST,
                    stroke_width=1.5, fill_opacity=0,
                    stroke_opacity=0.5,
                ).move_to(wec.get_center())
                ripples.append(ripple)

            ripple_anims = []
            for r in ripples:
                ripple_anims.append(
                    r.animate.scale(10).set_stroke(opacity=0),
                )

            self.play(
                wave_phase.animate(rate_func=linear).increment_value(2.0),
                *ripple_anims,
                run_time=1.5,
            )
            for r in ripples:
                self.remove(r)

        wave_bands.remove_updater(update_waves)

        # Label
        alone_label = Text(
            "Scattered waves dissipate → each WEC acts alone",
            font_size=22, color=GRAY_B,
        )
        alone_label.to_edge(DOWN, buff=0.8)
        self.play(FadeIn(alone_label), run_time=0.5)

        # Power bars — lower fill
        power_group = self._draw_power_bars(wecs, target_fill=0.65, label="~65%")
        self.wait(1)

        # Cleanup
        all_act4 = VGroup(
            header, underline, ocean, wecs, wec_labels,
            brace, brace_lbl, wave_bands, alone_label, power_group,
        )
        self.play(FadeOut(all_act4), run_time=0.6)

    # ── Act 5: Comparison + Closing ───────────────────────

    def show_comparison(self):
        title = Text("Farm Output Comparison", font_size=38, color=WHITE)
        title.to_edge(UP, buff=0.6)
        self.play(Write(title), run_time=0.8)

        # Divider
        divider = DashedLine(UP * 1.5, DOWN * 1.5, color=GRAY, stroke_width=1)
        self.play(Create(divider), run_time=0.3)

        # ── Left: Tight ──
        tight_header = Text("Tight (~70 m)", font_size=26, color=CONSTRUCTIVE_COLOR)
        tight_header.move_to(LEFT * 3.2 + UP * 1.2)

        tight_bar_bg = Rectangle(
            width=0.8, height=3, color=GRAY,
            fill_opacity=0.1, stroke_width=1, stroke_color=GRAY_B,
        ).move_to(LEFT * 3.2 + DOWN * 0.3)

        tight_bar_fill = Rectangle(
            width=0.8, height=3 * 0.88, color=CONSTRUCTIVE_COLOR,
            fill_opacity=0.6, stroke_width=0,
        )
        tight_bar_fill.align_to(tight_bar_bg, DOWN)

        tight_value = Text("4.1 MW", font_size=24, color=CONSTRUCTIVE_COLOR, weight=BOLD)
        tight_value.next_to(tight_bar_bg, DOWN, buff=0.3)

        # ── Right: Spread ──
        spread_header = Text("Wide (~110 m)", font_size=26, color=WAVE_CREST)
        spread_header.move_to(RIGHT * 3.2 + UP * 1.2)

        spread_bar_bg = Rectangle(
            width=0.8, height=3, color=GRAY,
            fill_opacity=0.1, stroke_width=1, stroke_color=GRAY_B,
        ).move_to(RIGHT * 3.2 + DOWN * 0.3)

        spread_bar_fill = Rectangle(
            width=0.8, height=3 * 0.65, color=WAVE_CREST,
            fill_opacity=0.5, stroke_width=0,
        )
        spread_bar_fill.align_to(spread_bar_bg, DOWN)

        spread_value = Text("3.8 MW", font_size=24, color=WAVE_CREST, weight=BOLD)
        spread_value.next_to(spread_bar_bg, DOWN, buff=0.3)

        # Animate bars
        self.play(
            FadeIn(tight_header), FadeIn(spread_header),
            FadeIn(tight_bar_bg), FadeIn(spread_bar_bg),
            run_time=0.5,
        )
        self.play(
            FadeIn(tight_bar_fill, shift=UP * 0.3),
            FadeIn(spread_bar_fill, shift=UP * 0.3),
            run_time=1,
        )
        self.play(FadeIn(tight_value), FadeIn(spread_value), run_time=0.5)
        self.wait(0.5)

        # Difference arrow
        diff_arrow = Arrow(
            RIGHT * 1.5 + DOWN * 0.3, LEFT * 1.5 + DOWN * 0.3,
            color=YELLOW, stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
        )
        diff_label = Text("+8% power", font_size=22, color=YELLOW)
        diff_label.next_to(diff_arrow, UP, buff=0.15)
        self.play(GrowArrow(diff_arrow), FadeIn(diff_label), run_time=0.6)
        self.wait(0.5)

        # Key insight
        insight = Text(
            "Optimal spacing exploits constructive wave interference",
            font_size=26, color=YELLOW,
        )
        box = SurroundingRectangle(insight, color=YELLOW, buff=0.15)
        insight_group = VGroup(insight, box).to_edge(DOWN, buff=0.6)
        self.play(Write(insight), Create(box), run_time=1)
        self.wait(1)

        # Course footer
        course = Text(
            "CE49X — Computational Thinking & Data Science",
            font_size=18, color=GRAY_B,
        )
        course.to_edge(DOWN, buff=0.2)
        self.play(
            insight_group.animate.shift(UP * 0.3),
            FadeIn(course),
            run_time=0.5,
        )
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)

    # ── Helper: Power bars next to WECs ───────────────────

    def _draw_power_bars(self, wecs, target_fill, label):
        """Draw small vertical power bars next to each WEC."""
        bar_group = VGroup()
        bar_w = 0.2
        bar_h = 1.2

        for wec in wecs:
            pos = wec.get_center() + RIGHT * 0.45 + UP * 0.3

            bg = Rectangle(
                width=bar_w, height=bar_h,
                color=GRAY, fill_opacity=0.1,
                stroke_width=1, stroke_color=GRAY_B,
            ).move_to(pos)

            fill = Rectangle(
                width=bar_w, height=bar_h * target_fill,
                fill_opacity=0.7, stroke_width=0,
                color=interpolate_color(
                    ManimColor(POWER_LOW), ManimColor(POWER_COLOR), target_fill
                ),
            )
            fill.align_to(bg, DOWN)

            pct = Text(label, font_size=12, color=WHITE)
            pct.next_to(bg, UP, buff=0.1)

            bar_group.add(bg, fill, pct)

        self.play(
            LaggedStart(*[FadeIn(m) for m in bar_group], lag_ratio=0.05),
            run_time=0.8,
        )
        return bar_group
