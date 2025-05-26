import argparse
import argparse
import builtins

from src.utils.cli_tools import (
    run_tool,
    list_methods,
    help_function,
    handle_command,
    ask_user_for_metrics,
    valid_metrics
)


# tests for run_tool() and list_methods()
def test_run_tool_returns_expected_output():
    metrics = ['ppc', 'c2st']
    result = run_tool(metrics)
    assert result == 'Running the tool with metrics: ppc, c2st'


def test_list_methods_returns_methods_list():
    expected = "Method 1: ExampleMethodA\nMethod 2: ExampleMethodB"
    assert list_methods() == expected


# tests for handle_run_command()
def test_handle_run_command_with_valid_metrics(monkeypatch, capsys):
    args = argparse.Namespace(metrics='ppc,c2st')
    # No interactive input needed
    monkeypatch.setattr(builtins, 'input', lambda *args, **kwargs: '')

    handle_command(args)
    captured = capsys.readouterr()
    assert 'Running the tool with metrics: ppc, c2st' in captured.out


def test_handle_run_command_with_invalid_metrics(monkeypatch, capsys):
    args = argparse.Namespace(metrics='ppc,nonsense')
    monkeypatch.setattr(builtins, 'input', lambda *args, **kwargs: '')

    handle_command(args)
    captured = capsys.readouterr()
    assert 'Invalid metrics: nonsense' in captured.out
    assert 'Valid metrics are:' in captured.out


# test ask_user_for_metrics()
def test_ask_user_for_metrics(monkeypatch):
    inputs = iter(['y', 'y', 'y', 'y'])
    monkeypatch.setattr(builtins, 'input', lambda *args, **kwargs: next(inputs))
    result = ask_user_for_metrics()
    assert result == valid_metrics










