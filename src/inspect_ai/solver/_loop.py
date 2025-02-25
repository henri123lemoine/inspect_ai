from ._solver import Generate, Solver, solver
from ._task_state import TaskState


@solver
def loop(
    solver: Solver,
    condition: Solver | None = None,
    max_iterations: int = 10,
) -> Solver:
    """Create a solver that repeatedly applies another solver until a condition is satisfied.

    The solver will run for a maximum number of iterations or until either the condition
    is met or the state is marked as completed.

    Args:
        solver: The solver to be applied repeatedly.
        condition: Optional solver that checks the state. If it marks the state as completed,
            the loop terminates early. If the solver returns None (e.g. when chaining empty solvers),
            the condition is considered not met.
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
        condition: Optional solver that checks the state. If it marks the state as completed,
            the loop terminates early. If the solver returns None (e.g. when chaining empty solvers),
            the condition is considered not met.
        max_iterations: Maximum number of iterations.
    """

    def __init__(
        self,
        solver: Solver,
        condition: Solver | None = None,
        max_iterations: int = 10,
    ) -> None:
        self._solver = solver
        self._condition = condition
        self._max_iterations = max_iterations

    async def __call__(self, state: TaskState, generate: Generate) -> TaskState:
        from ._transcript import solver_transcript

        for _ in range(self._max_iterations):
            # Run the main solver
            with solver_transcript(self._solver, state, "loop") as st:
                state = await self._solver(state, generate)
                st.complete(state)

            assert state, "Loop solver returned None"

            # Check if we should stop
            if state.completed:
                break

            # Run the condition solver if provided
            if self._condition is not None:
                with solver_transcript(self._condition, state, "loop_condition") as st:
                    condition_state = await self._condition(state, generate)
                    st.complete(condition_state)
                # Only check completion if condition_state is not None
                if condition_state is not None and condition_state.completed:
                    break

        assert state is not None, "Loop solver returned None"

        return state
