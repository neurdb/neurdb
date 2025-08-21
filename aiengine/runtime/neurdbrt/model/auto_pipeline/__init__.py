def neurdb_on_start():
    from neurdbrt.model import register_model

    from .builder import AutoPipelineBuilder

    register_model("auto_pipeline", AutoPipelineBuilder)
