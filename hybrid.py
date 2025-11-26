from vespa.package import (
    ApplicationPackage,
    Document,
    Field,
    FieldSet,
    Function,
    RankProfile,
    Schema,
    HNSW,
    GlobalPhaseRanking,
)
from vespa.deployment import VespaDocker
from vespa.io import VespaResponse
from datasets import load_dataset
from tqdm import tqdm

package = ApplicationPackage(
    name="hybridsearch",
    schema=[
        Schema(
            name="doc",
            document=Document(
                fields=[
                    Field(
                        name="id",
                        type="string",
                        indexing=["summary"],
                    ),
                    Field(
                        name="text",
                        type="string",
                        indexing=["index", "summary"],
                        index="enable-bm25",
                    ),
                    Field(
                        name="url",
                        type="string",
                        indexing=["index", "summary"],
                        index="enable-bm25",
                    ),
                    Field(
                        name="text_embedding",
                        type="tensor(x[384])",
                        indexing=["index", "attribute"],
                        ann=HNSW(distance_metric="angular"),
                    ),
                ]
            ),
            fieldsets=[
                FieldSet(name="default", fields=["text", "url"]),
            ],
            rank_profiles=[
                RankProfile(
                    name="bm25",
                    inputs=[("query(q)",)],
                    functions=[
                        Function(
                            name="bm25texturl",
                            expression="bm25(text) + 0.1 * bm25(url)",
                        ),
                    ],
                    first_phase="bm25texturl",
                ),
                RankProfile(
                    name="semantic",
                    inputs=[("query(q)", "tensor(x[384])")],
                    first_phase="closeness(field, text_embedding)",
                ),
                RankProfile(
                    name="fusion",
                    inherits="bm25",
                    inputs=[("query(q)", "tensor(x[384])")],
                    first_phase="closeness(field, text_embedding)",
                    global_phase=GlobalPhaseRanking(
                        expression="reciprocal_rank_fusion(bm25texturl, closeness(field, text_embedding))",
                        rerank_count=1000,
                    ),
                ),
            ],
        ),
    ],
)
