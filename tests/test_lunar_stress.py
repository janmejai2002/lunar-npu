import pytest
from lunar_core.engine import LunarNPUEngine
from lunar_core.stress import (
    LunarStressEngine,
    build_systolic_stress_model,
    run_npu_stress_test,
)


def test_build_systolic_stress_model():
    model = build_systolic_stress_model(batch=32, d_in=256, d_hidden=512, d_out=256)
    assert model is not None
    assert len(model.inputs) == 1
    assert len(model.outputs) == 1
    assert model.inputs[0].get_shape() == [32, 256]
    assert model.outputs[0].get_shape() == [32, 256]


def test_lunar_stress_engine_execution():
    engine = LunarNPUEngine()
    stress = LunarStressEngine(engine=engine)
    res = stress.run_stress(iterations=5)

    assert "device" in res
    assert "is_npu" in res
    assert res["iterations"] == 5
    assert res["duration_ms"] > 0
    assert "sustained_tflops" in res
    assert "effective_int8_tops" in res
    assert "package_power_w" in res
    assert "temperature_c" in res
    assert res["flops_per_inference"] > 0
    assert res["total_gigaflops_executed"] > 0


def test_run_npu_stress_test_convenience():
    res = run_npu_stress_test(iterations=2)
    assert isinstance(res, dict)
    assert res["iterations"] == 2
    assert res["duration_ms"] > 0
