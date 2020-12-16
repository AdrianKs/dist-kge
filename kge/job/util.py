import numpy as np
import torch
from torch import Tensor
from kge.indexing import intersection


def get_sp_po_coords_from_spo_batch(
    batch: Tensor, num_entities: int, sp_index: dict, po_index: dict, targets: np.array
) -> torch.Tensor:
    """Given a set of triples , lookup matches for (s,p,?) and (?,p,o).

    Each row in batch holds an (s,p,o) triple. Returns the non-zero coordinates
    of a 2-way binary tensor with one row per triple and 2*num_entites columns.
    The first half of the columns correspond to hits for (s,p,?); the second
    half for (?,p,o).

    """
    num_ones = 0
    NOTHING = torch.zeros([0], dtype=torch.long)
    if targets is None:
        for i, triple in enumerate(batch):
            s, p, o = triple[0].item(), triple[1].item(), triple[2].item()
            num_ones += len(sp_index.get((s, p), NOTHING))
            num_ones += len(po_index.get((p, o), NOTHING))

        coords = torch.empty([num_ones, 2], dtype=torch.long)
    else:
        coords = torch.empty([len(batch) * len(targets), 2], dtype=torch.long)
    current_index = 0
    for i, triple in enumerate(batch):
        s, p, o = triple[0].item(), triple[1].item(), triple[2].item()

        objects = sp_index.get((s, p), NOTHING)
        subjects = po_index.get((p, o), NOTHING) + num_entities
        if targets is None:
            relevant_objects = objects
            relevant_subjects = subjects
        else:
            relevant_objects = torch.from_numpy(intersection(objects.numpy(), targets))
            relevant_subjects = torch.from_numpy(
                intersection(subjects.numpy(), targets)
            )
        coords[current_index : (current_index + len(relevant_objects)), 0] = i
        coords[
            current_index : (current_index + len(relevant_objects)), 1
        ] = relevant_objects
        current_index += len(relevant_objects)

        coords[current_index : (current_index + len(relevant_subjects)), 0] = i
        coords[
            current_index : (current_index + len(relevant_subjects)), 1
        ] = relevant_subjects
        current_index += len(relevant_subjects)

    return coords[:current_index]


def coord_to_sparse_tensor(
    nrows: int, ncols: int, coords: Tensor, device: str, value=1.0, row_slice=None
):
    if row_slice is not None:
        if row_slice.step is not None:
            # just to be sure
            raise ValueError()

        coords = coords[
            (coords[:, 0] >= row_slice.start) & (coords[:, 0] < row_slice.stop), :
        ]
        coords[:, 0] -= row_slice.start
        nrows = row_slice.stop - row_slice.start

    if device == "cpu":
        labels = torch.sparse.FloatTensor(
            coords.long().t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
        )
    else:
        labels = torch.cuda.sparse.FloatTensor(
            coords.long().t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
            device=device,
        )

    return labels
