"""
Playback Controller Module

This module provides the PlaybackController class for managing time animation
playback with play/pause, step forward/backward, and speed control.
"""

import logging
import param
import panel as pn

logger = logging.getLogger(__name__)


# =============================================================================
# Playback Controller
# =============================================================================

class PlaybackController(param.Parameterized):
    """
    Controller for time animation playback.

    This controller manages:
    - Play/Pause functionality
    - Step forward/backward through time
    - Playback speed control
    - Session management to prevent zombie loops

    The controller uses a session ID mechanism to ensure that only one
    playback loop is active at a time, preventing issues when users
    rapidly click play/pause.

    Attributes
    ----------
    playing : bool
        Whether playback is currently active
    session_id : int
        Current session ID (incremented on each play/pause to kill old loops)

    Parameters
    ----------
    time_slider : pn.widgets.DiscreteSlider
        Time slider widget to control
    play_button : pn.widgets.Button
        Play/pause button
    prev_button : pn.widgets.Button, optional
        Previous frame button
    next_button : pn.widgets.Button, optional
        Next frame button
    speed_slider : pn.widgets.IntSlider, optional
        Playback speed control (milliseconds between frames)

    Examples
    --------
    >>> time_slider = pn.widgets.DiscreteSlider(name='Time', options={'0': 0, '1': 1})
    >>> play_btn = pn.widgets.Button(name='▶')
    >>> controller = PlaybackController(time_slider, play_btn)
    >>> controller.toggle_play(None)  # Start playing
    >>> controller.stop()  # Stop playing
    """

    playing = param.Boolean(default=False, doc="Whether playback is active")
    session_id = param.Integer(default=0, doc="Session ID to prevent zombie loops")

    def __init__(
        self,
        time_slider,
        play_button,
        prev_button=None,
        next_button=None,
        speed_slider=None,
        **params
    ):
        """
        Initialize PlaybackController.

        Parameters
        ----------
        time_slider : pn.widgets.DiscreteSlider
            Time slider widget
        play_button : pn.widgets.Button
            Play/pause button
        prev_button : pn.widgets.Button, optional
            Previous frame button
        next_button : pn.widgets.Button, optional
            Next frame button
        speed_slider : pn.widgets.IntSlider, optional
            Speed control slider (ms)
        """
        super().__init__(**params)

        self.time_slider = time_slider
        self.play_button = play_button
        self.prev_button = prev_button
        self.next_button = next_button
        self.speed_slider = speed_slider

        # Connect button callbacks
        self.play_button.on_click(self.toggle_play)

        if self.prev_button is not None:
            self.prev_button.on_click(self.step_backward)

        if self.next_button is not None:
            self.next_button.on_click(self.step_forward)

    def toggle_play(self, event):
        """
        Toggle between play and pause states.

        When starting playback:
        - Changes button icon to pause (⏸)
        - Increments session ID to kill any previous loops
        - Starts periodic callback for stepping through frames

        When pausing:
        - Changes button icon to play (▶)
        - Increments session ID to stop current loop

        Parameters
        ----------
        event : param.Event
            Button click event
        """
        self.playing = not self.playing

        if self.playing:
            # Start playing
            self.play_button.name = '⏸'
            self.session_id += 1
            current_id = self.session_id

            # Start first step immediately
            pn.state.add_periodic_callback(
                lambda: self._step_internal(current_id),
                period=10,
                count=1
            )

            logger.debug(f"Playback started (session {current_id})")

        else:
            # Pause
            self.play_button.name = '▶'
            self.session_id += 1  # Kill any pending loops
            logger.debug("Playback paused")

    def _step_internal(self, run_id: int):
        """
        Internal step function called by periodic callback.

        This function:
        1. Checks if this loop is still valid (session guard)
        2. Checks if still playing
        3. Advances slider to next time step
        4. Schedules next step if still playing

        Parameters
        ----------
        run_id : int
            Session ID for this playback loop
        """
        # Session guard: kill zombie loops
        if run_id != self.session_id:
            logger.debug(f"Killing zombie loop (session {run_id})")
            return

        # Check if still playing
        if not self.playing:
            return

        # Check if slider is available
        if not self.time_slider.visible:
            logger.debug("Time slider not visible, stopping playback")
            self.stop()
            return

        # Get available time options
        options = list(self.time_slider.options.values())
        if not options:
            logger.warning("No time options available")
            self.stop()
            return

        # Advance to next time step
        current = self.time_slider.value
        try:
            idx = options.index(current)
            next_idx = (idx + 1) % len(options)
            self.time_slider.value = options[next_idx]
            logger.debug(f"Advanced to time index {next_idx}")
        except ValueError:
            # Current value not in options, reset to start
            self.time_slider.value = options[0]
            logger.debug("Reset to first time step")

        # Schedule next step (with same run_id to maintain session)
        if self.playing:
            speed = self.speed_slider.value if self.speed_slider else 1000
            pn.state.add_periodic_callback(
                lambda: self._step_internal(run_id),
                period=speed,
                count=1
            )

    def stop(self):
        """
        Stop playback.

        Updates button icon and increments session ID to kill pending loops.
        """
        self.playing = False
        self.play_button.name = '▶'
        self.session_id += 1
        logger.debug("Playback stopped")

    def step_forward(self, event=None):
        """
        Step forward one time frame.

        Parameters
        ----------
        event : param.Event, optional
            Button click event
        """
        if not self.time_slider.visible:
            return

        options = list(self.time_slider.options.values())
        if not options:
            return

        current = self.time_slider.value
        try:
            idx = options.index(current)
            next_idx = (idx + 1) % len(options)
            self.time_slider.value = options[next_idx]
            logger.debug(f"Stepped forward to index {next_idx}")
        except ValueError:
            self.time_slider.value = options[0]

    def step_backward(self, event=None):
        """
        Step backward one time frame.

        Parameters
        ----------
        event : param.Event, optional
            Button click event
        """
        if not self.time_slider.visible:
            return

        options = list(self.time_slider.options.values())
        if not options:
            return

        current = self.time_slider.value
        try:
            idx = options.index(current)
            prev_idx = (idx - 1) % len(options)
            self.time_slider.value = options[prev_idx]
            logger.debug(f"Stepped backward to index {prev_idx}")
        except ValueError:
            self.time_slider.value = options[-1]


# =============================================================================
# Convenience Functions
# =============================================================================

def create_playback_controller(
    time_slider,
    play_button,
    prev_button=None,
    next_button=None,
    speed_slider=None
) -> PlaybackController:
    """
    Create and configure a PlaybackController.

    This is a convenience function that instantiates a PlaybackController
    with the provided widgets.

    Parameters
    ----------
    time_slider : pn.widgets.DiscreteSlider
        Time slider widget
    play_button : pn.widgets.Button
        Play/pause button
    prev_button : pn.widgets.Button, optional
        Previous frame button
    next_button : pn.widgets.Button, optional
        Next frame button
    speed_slider : pn.widgets.IntSlider, optional
        Speed control slider

    Returns
    -------
    PlaybackController
        Configured playback controller

    Examples
    --------
    >>> from vvmviz.ui.widgets import create_time_controls
    >>> controls = create_time_controls()
    >>> controller = create_playback_controller(
    ...     time_slider=controls['slider'],
    ...     play_button=controls['play'],
    ...     prev_button=controls['prev'],
    ...     next_button=controls['next'],
    ...     speed_slider=controls['speed']
    ... )
    """
    return PlaybackController(
        time_slider=time_slider,
        play_button=play_button,
        prev_button=prev_button,
        next_button=next_button,
        speed_slider=speed_slider
    )
