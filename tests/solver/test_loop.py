from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.log._log import EvalLog
from inspect_ai.solver import Generate, Solver, TaskState, chain, loop, solver


@solver
def increase_counter() -> Solver:
    """A simple solver that increments a counter stored in the TaskState."""

    async def solve(state: TaskState, _generate: Generate) -> TaskState:
        counter = state.store.get("counter", 0)
        state.store.set("counter", counter + 1)
        return state

    return solve


@solver
def complete_immediately() -> Solver:
    """A solver that increments the counter and then marks the state as complete.

    This should cause loop termination after one iteration.
    """

    async def solve(state: TaskState, _generate: Generate) -> TaskState:
        counter = state.store.get("counter", 0)
        state.store.set("counter", counter + 1)
        state.completed = True
        return state

    return solve


@solver
def check_counter_three() -> Solver:
    """A condition solver that completes when counter >= 3."""

    async def solve(state: TaskState, _generate: Generate) -> TaskState:
        if state.store.get("counter", 0) >= 3:
            state.completed = True
        return state

    return solve


@solver
def never_complete() -> Solver:
    """A condition solver that never completes."""

    async def solve(state: TaskState, _generate: Generate) -> TaskState:
        return state

    return solve


@task
def loop_condition_task() -> Task:
    """Test that the loop stops once the counter in the state store reaches 3."""

    @solver
    def loop_solver() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            # Create a loop solver that runs 'increase_counter'
            loop_solver_instance = loop(
                solver=increase_counter(),
                condition=check_counter_three(),
                max_iterations=10,
            )
            state = await loop_solver_instance(state, generate)
            # Verify that the counter is exactly 3.
            assert state.store.get("counter", 0) == 3, (
                f"Expected counter to be 3, got {state.store.get('counter', 0)}"
            )
            return state

        return solve

    return Task(
        dataset=[Sample(input="dummy", target="dummy")],
        solver=loop_solver(),
    )


@task
def loop_max_iterations_task() -> Task:
    """Test that if the stop condition is never met, the loop runs for exactly max_iterations iterations."""

    @solver
    def loop_solver() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            loop_solver_instance = loop(
                solver=increase_counter(),
                condition=never_complete(),  # Never stops early
                max_iterations=5,
            )
            state = await loop_solver_instance(state, generate)
            assert state.store.get("counter", 0) == 5, (
                f"Expected counter to be 5, got {state.store.get('counter', 0)}"
            )
            return state

        return solve

    return Task(
        dataset=[Sample(input="dummy", target="dummy")],
        solver=loop_solver(),
    )


@task
def loop_completion_task() -> Task:
    """Test that the loop stops immediately when the solver flags state.completed."""

    @solver
    def loop_solver() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            loop_solver_instance = loop(
                solver=complete_immediately(),
                condition=never_complete(),  # Condition not used since .completed wins
                max_iterations=10,
            )
            state = await loop_solver_instance(state, generate)
            # The counter should be 1 because complete_immediately marks the state as completed.
            assert state.store.get("counter", 0) == 1, (
                f"Expected counter to be 1, got {state.store.get('counter', 0)}"
            )
            return state

        return solve

    return Task(
        dataset=[Sample(input="dummy", target="dummy")],
        solver=loop_solver(),
    )


@task
def loop_evaluation_task() -> Task:
    """Test that simply evaluating the loop() solver does not produce an error and works as expected."""

    @solver
    def loop_solver() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            # Although the condition solver never completes, complete_immediately() marks the state as completed,
            # so the loop should exit immediately after one iteration.
            loop_solver_instance = loop(
                solver=complete_immediately(),
                condition=never_complete(),
                max_iterations=3,
            )
            state = await loop_solver_instance(state, generate)
            # Expected: counter is incremented exactly once.
            assert state.store.get("counter", 0) == 1, (
                f"Expected counter to be 1, got {state.store.get('counter', 0)}"
            )
            return state

        return solve

    return Task(
        dataset=[Sample(input="dummy", target="dummy")],
        solver=loop_solver(),
    )


@task
def chain_loop_task() -> Task:
    """Test that chaining a loop solver works as expected."""

    @solver
    def chain_solver() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            # Wrap the loop solver in a chain.
            chained_solver = chain(
                loop(
                    solver=complete_immediately(),
                    condition=never_complete(),
                    max_iterations=3,
                )
            )
            state = await chained_solver(state, generate)
            # complete_immediately() should run just once.
            assert state.store.get("counter", 0) == 1, (
                f"Expected counter to be 1, got {state.store.get('counter', 0)}"
            )
            return state

        return solve

    return Task(
        dataset=[Sample(input="dummy", target="dummy")],
        solver=chain_solver(),
    )


@task
def chain_of_solvers_task() -> Task:
    """Test that chaining a normal solver with a loop solver works together."""

    @solver
    def chain_solver() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            # Chain a simple solver and then a loop solver.
            chained_solver = chain(
                increase_counter(),
                loop(
                    solver=increase_counter(),
                    condition=check_counter_three(),
                    max_iterations=5,
                ),
            )
            state = await chained_solver(state, generate)
            # Expected behavior:
            #   - The first increase_counter increments counter from 0 to 1.
            #   - The loop then runs increase_counter until counter >= 3.
            #     It should run two iterations: 1 -> 2 and 2 -> 3.
            assert state.store.get("counter", 0) == 3, (
                f"Expected counter to be 3, got {state.store.get('counter', 0)}"
            )
            return state

        return solve

    return Task(
        dataset=[Sample(input="dummy", target="dummy")],
        solver=chain_solver(),
    )


@task
def direct_loop_task() -> Task:
    """
    Test that a solver defined directly from loop() is passable to eval().

    Uses a loop on complete_immediately(), where the loop should exit immediately
    due to the state.completed flag, even though the condition never completes.

    Expected: counter is incremented exactly once.
    """
    return Task(
        dataset=[Sample(input="dummy", target="dummy")],
        solver=loop(
            solver=complete_immediately(),
            condition=never_complete(),
            max_iterations=3,
        ),
    )


@task
def direct_chain_task() -> Task:
    """
    Test that a solver defined directly from chain() is passable to eval().

    Chains increase_counter() and complete_immediately().
    Expected behavior:
      - increase_counter() increments counter from 0 to 1.
      - complete_immediately() increments it further (1 to 2) and marks state.completed.
    """
    return Task(
        dataset=[Sample(input="dummy", target="dummy")],
        solver=chain(increase_counter(), complete_immediately()),
    )


@task
def direct_chain_loop_task() -> Task:
    """
    Test that chaining a regular solver with a loop solver directly is evaluated correctly.

    Chains a single call to increase_counter() with a loop that repeats increase_counter()
    until the counter in the state reaches 3.

    Expected behavior:
      - The chained increase_counter() (outside the loop) bumps counter: 0 -> 1.
      - The loop then runs increase_counter() until counter >= 3.
        With counter starting at 1, it takes two iterations: 1 -> 2 and 2 -> 3.
    """
    return Task(
        dataset=[Sample(input="dummy", target="dummy")],
        solver=chain(
            increase_counter(),
            loop(
                solver=increase_counter(),
                condition=check_counter_three(),
                max_iterations=5,
            ),
        ),
    )


def _assert_counter_in_logs(logs: list[EvalLog], expected: int) -> None:
    for log in logs:
        if log.samples is None:
            continue
        for sample in log.samples:
            counter = sample.store.get("counter", 0)
            assert counter == expected, f"Expected counter {expected}, got {counter}"


def test_loop_condition_store() -> None:
    """Verify that loop_condition_task produces samples with counter==3."""
    logs = eval(loop_condition_task(), model="mockllm/model")
    _assert_counter_in_logs(logs, expected=3)


def test_loop_max_iterations_store() -> None:
    """Verify that loop_max_iterations_task produces samples with counter==5."""
    logs = eval(loop_max_iterations_task(), model="mockllm/model")
    _assert_counter_in_logs(logs, expected=5)


def test_loop_completion_store() -> None:
    """Verify that loop_completion_task produces samples with counter==1."""
    logs = eval(loop_completion_task(), model="mockllm/model")
    _assert_counter_in_logs(logs, expected=1)


def test_loop_evaluation_store() -> None:
    """Verify that loop_evaluation_task produces samples with counter==1."""
    logs = eval(loop_evaluation_task(), model="mockllm/model")
    _assert_counter_in_logs(logs, expected=1)


def test_chain_loop_store() -> None:
    """Verify that chain_loop_task produces samples with counter==1."""
    logs = eval(chain_loop_task(), model="mockllm/model")
    _assert_counter_in_logs(logs, expected=1)


def test_chain_of_solvers_store() -> None:
    """Verify that chain_of_solvers_task produces samples with counter==3."""
    logs = eval(chain_of_solvers_task(), model="mockllm/model")
    _assert_counter_in_logs(logs, expected=3)


def test_direct_loop_store() -> None:
    """Verify that direct_loop_task produces samples with counter==1."""
    logs = eval(direct_loop_task(), model="mockllm/model")
    _assert_counter_in_logs(logs, expected=1)


def test_direct_chain_store() -> None:
    """Verify that direct_chain_task produces samples with counter==2."""
    logs = eval(direct_chain_task(), model="mockllm/model")
    _assert_counter_in_logs(logs, expected=2)


def test_direct_chain_loop_store() -> None:
    """Verify that direct_chain_loop_task produces samples with counter==3."""
    logs = eval(direct_chain_loop_task(), model="mockllm/model")
    _assert_counter_in_logs(logs, expected=3)
