from typing import Callable

from ._solver import Generate, Solver, solver
from ._task_state import TaskState


@solver
def loop(
    solver: Solver,
    condition: Callable[[TaskState], bool] | None = None,
    max_iterations: int = 10,
) -> Solver:
    """Create a solver that repeatedly applies another solver until a condition is satisfied.

    The solver will run for a maximum number of iterations or until either the condition
    is met or the state is marked as completed.

    Args:
        solver: The solver to be applied repeatedly.
        condition: Optional callable that checks the state. If it returns True, the loop terminates early.
        max_iterations: Maximum number of iterations.

    Returns:
        A solver that executes the given solver repeatedly in a loop.
    """
    return Loop(solver, condition, max_iterations)


class Loop(Solver):
    """A solver that repeatedly applies another solver until a condition is met.

    The solver will run for a maximum number of iterations or until either the condition
    is met or the state is marked as completed.

    Args:
        solver: The solver to be applied repeatedly.
        condition: Optional callable that checks the state. If it returns True, the loop terminates early.
        max_iterations: Maximum number of iterations.
    """

    def __init__(
        self,
        solver: Solver,
        condition: Callable[[TaskState], bool] | None = None,
        max_iterations: int = 10,
    ) -> None:
        self._solver = solver
        self._condition = condition
        self._max_iterations = max_iterations

    async def __call__(self, state: TaskState, generate: Generate) -> TaskState:
        from ._transcript import solver_transcript

        for _ in range(self._max_iterations):
            with solver_transcript(self._solver, state, "loop") as st:
                state = await self._solver(state, generate)
                st.complete(state)
            if (
                self._condition is not None and self._condition(state)
            ) or state.completed:
                break
        return state
