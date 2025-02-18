from typing import Callable

from ._chain import Chain
from ._solver import Generate, Solver
from ._task_state import TaskState


def loop(
    solver: Solver,
    condition: Callable[[TaskState], bool] | None = None,
    max_iterations: int = 10,
) -> Solver:
    """Repeatedly applies a solver until a condition is satisfied.

    The solver will run for a maximum number of iterations or until either the condition
    is met or the state is marked as completed.

    Args:
        solver: The solver to be applied repeatedly.
        condition: Optional callable that checks the state. If it returns True, the loop terminates early.
        max_iterations: The maximum number of iterations to execute.

    Returns:
        A solver that implements the looping behavior.
    """
    return Loop([solver] * max_iterations, condition)


class Loop(Chain):
    """A solver that repeatedly applies another solver until a condition is met.

    Args:
        solver: The solver to be applied repeatedly.
        condition: Optional callable that checks the state. If it returns True, the loop terminates early.
        max_iterations: The maximum number of iterations to execute.
    """

    def __init__(
        self,
        solvers: list[Solver],
        condition: Callable[[TaskState], bool] | None = None,
    ) -> None:
        self._solvers = solvers
        self._condition = condition

    async def __call__(
        self,
        state: TaskState,
        generate: Generate,
    ) -> TaskState:
        from ._transcript import solver_transcript

        for solver in self._solvers:
            with solver_transcript(solver, state, "loop") as st:
                state = await solver(state, generate)
                st.complete(state)
            if (
                self._condition is not None
                and self._condition(state)
                or state.completed
            ):
                break

        return state
