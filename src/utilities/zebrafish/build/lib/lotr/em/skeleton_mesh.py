import numpy as np
import trimesh


def _concatenate_meshes(mesh_list):
    """Concatenate list of separate meshes."""

    # Find number of vertices and edges:
    n_vert = np.zeros(len(mesh_list), dtype=int)
    n_faces = np.zeros(len(mesh_list), dtype=int)

    for i, mesh in enumerate(mesh_list):
        n_vert[i] = mesh.vertices.shape[0]
        n_faces[i] = mesh.faces.shape[0]

    # Instantiate new vectors...
    vertices = np.zeros((n_vert.sum(), 3))
    faces = np.zeros((n_faces.sum(), 3))
    face_normals = np.zeros((n_faces.sum(), 3))

    # ...keep count of vertices and faces...
    cum_vert = 0
    cum_faces = 0
    # ...and loop and fill
    for i, mesh in enumerate(mesh_list):
        vertices[cum_vert : cum_vert + n_vert[i], :] = mesh.vertices
        faces[cum_faces : cum_faces + n_faces[i], :] = mesh.faces + cum_vert
        face_normals[cum_faces : cum_faces + n_faces[i], :] = mesh.face_normals

        # Update counts:
        cum_vert += n_vert[i]
        cum_faces += n_faces[i]

    return trimesh.Trimesh(vertices=vertices, faces=faces, face_normals=face_normals)


def make_cylinder_tree(coords, edges, n_sections=7, radius=4):
    """Create mesh of concatenated cylinders. Bottleneck of this function
    is generating so many cylinders with trimesh, little room for optimizaiton.
    """

    cyl_list = []
    for i, seg in enumerate(edges):
        # Generate cylinder:
        cyl = trimesh.creation.cylinder(
            radius=radius,
            height=0,
            segment=(coords[seg[0], :], coords[seg[1], :]),
            sections=n_sections,
        )
        cyl_list.append(cyl)

    return _concatenate_meshes(cyl_list)


def make_full_neuron(
    coords,
    soma_idx,
    dendr_edges,
    axon_edges,
    n_sections=7,
    axon_radius=4,
    dendrite_radius=4,
    soma_radius=100,
    soma_subdivisions=4,
):
    """Function to create a 3D mesh from a tracing tree."""

    # Create sphere for the soma at seed position:
    soma_mesh = trimesh.creation.icosphere(
        radius=soma_radius, subdivisions=soma_subdivisions
    )

    # Translate to soma location:
    soma_mesh.vertices = soma_mesh.vertices + coords[soma_idx, :]

    # Create cylinder tree for the dendrites:
    dendrite_mesh = make_cylinder_tree(
        coords, dendr_edges, n_sections=n_sections, radius=dendrite_radius
    )

    # Create cylinder tree for the axon:
    axon_mesh = make_cylinder_tree(
        coords, axon_edges, n_sections=n_sections, radius=axon_radius
    )

    return _concatenate_meshes([soma_mesh, dendrite_mesh, axon_mesh])
