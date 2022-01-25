import numpy as np

class FileOperations():
    def __init__(self):
        print("here")

    @staticmethod
    def poly_write(file_name: str, nodes, faces, regions: dict, boundaries=None):
        with open(file_name, "wb") as m_file:
            # Nodes
            m_file.write("{} 3\n".format(len(nodes)).encode("utf-8"))
            for index in range(0, len(nodes)):
                m_file.write("{} {} {} {}\n".format(index + 1, *nodes[index]).encode("utf-8"))

            # Faces
            if boundaries is not None:
                m_file.write("{} 1\n".format(len(faces)).encode("utf-8"))
                for index in range(0, len(faces)):
                    m_file.write("1 0 {}\n".format(int(boundaries[index] + 1)).encode("utf-8"))
                    m_file.write("3 {} {} {}\n".format(*faces[index] + 1).encode("utf-8"))
            else:
                m_file.write("{} 0\n".format(len(faces)).encode("utf-8"))
                for index in range(0, len(faces)):
                    m_file.write("1 0\n".encode("utf-8"))
                    m_file.write("3 {} {} {}\n".format(*faces[index] + 1).encode("utf-8"))
                

            # Holes NOT IMPLEMENTED YET
            m_file.write("0\n".encode("utf-8"))

            # Regions
            m_file.write("{}\n".format(len(list(regions.keys()))).encode("utf-8"))
            for region_id in regions.keys():
                m_file.write("{} {} {} {} {} {}\n".format(region_id, *regions[region_id]['coordinates'], region_id, regions[region_id]['max_volume']).encode("utf-8"))

    @staticmethod
    def gmsh_write(file_name: str, surfaces, domains, physical_tags, bounding_surface_tag):
        c_size_t = np.dtype("P")

        with open(file_name, "wb") as m_file:
            # MeshFormat (Check)
            m_file.write(b"$MeshFormat\n")
            m_file.write("4.1 {} {}\n".format(1, c_size_t.itemsize).encode("utf-8"))

            m_file.write(b"$EndMeshFormat\n")

            # PhysicalNames (Check)
            m_file.write(b"$PhysicalNames\n")
            m_file.write("{}\n".format(len(physical_tags)).encode("utf-8"))
            for physical_tag in physical_tags:
                m_file.write('3 {} "{}"\n'.format(*physical_tag).encode("utf-8"))
            m_file.write(b"$EndPhysicalNames\n")

            # Entities (Check)
            m_file.write(b"$Entities\n")
            
            m_file.write("0 0 {} {}\n".format(len(surfaces), len(domains)).encode("utf-8"))
            for i in range(0, len(surfaces)):
                m_file.write("{} {} {} {} {} {} {} {} {}\n".format(i + 1, np.amin(surfaces[i].vertices[:, 0]), np.amin(surfaces[i].vertices[:, 1]), np.amin(surfaces[i].vertices[:, 2]), np.amax(surfaces[i].vertices[:, 0]), np.amax(surfaces[i].vertices[:, 1]), np.amax(surfaces[i].vertices[:, 2]), 0, 0).encode("utf-8"))
            for i in range(0, len(domains)):
                m_file.write("{} {} {} {} {} {} {} {} {} {} {}\n".format(i + 1, np.amin(domains[i].vertices[:, 0]), np.amin(domains[i].vertices[:, 1]), np.amin(domains[i].vertices[:, 2]), np.amax(domains[i].vertices[:, 0]), np.amax(domains[i].vertices[:, 1]), np.amax(domains[i].vertices[:, 2]), 1, physical_tags[i][0], 1, bounding_surface_tag[i]).encode("utf-8"))
            m_file.write(b"$EndEntities\n")

            # Nodes (Checked?)
            float_fmt=".16e"

            surfs = 0
            voxels = 0
            for surf in surfaces:
                surfs = surfs + surf.num_vertices
            for domain in domains:
                voxels = voxels + domain.num_vertices

            m_file.write(b"$Nodes\n")
            num_blocks = len(surfaces) + len(domains)
            min_tag = 1
            max_tag = surfs + voxels

            m_file.write("{} {} {} {}\n".format(num_blocks, surfs + voxels, min_tag, max_tag).encode("utf-8"))

            ta = 1
            face_elements = []
            for i in range(0, len(surfaces)):
                # dim_entity, is_parametric
                num_elem = surfaces[i].num_vertices

                m_file.write("{} {} {} {}\n".format(2, i + 1, 0, num_elem).encode("utf-8"))
                np.arange(ta, ta + num_elem).tofile(m_file, "\n", "%d")
                m_file.write(b"\n")
                np.savetxt(m_file, surfaces[i].vertices, delimiter=" ", fmt="%" + float_fmt)
                #face_elements.append(surfaces[i].faces + 1 + (ta - 1))
                face_elements.append(surfaces[i].faces.astype(c_size_t) + 1 + (ta - 1))
                ta = ta + num_elem
                m_file.write(b"\n")

            voxel_elements = []
            for i in range(0, len(domains)):
                num_elem = domains[i].num_vertices

                m_file.write("{} {} {} {}\n".format(3, i + 1, 0, num_elem).encode("utf-8"))
                np.arange(ta, ta + num_elem).tofile(m_file, "\n", "%d")
                m_file.write(b"\n")
                np.savetxt(m_file, domains[i].vertices, delimiter=" ", fmt="%" + float_fmt)
                #voxel_elements.append(domains[i].voxels + 1 + (ta - 1))
                voxel_elements.append(domains[i].voxels.astype(c_size_t) + 1 + (ta - 1))
                ta = ta + num_elem
                m_file.write(b"\n")
            m_file.write(b"$EndNodes\n")

            # Elements
            surfs = 0
            voxels = 0
            for surf in surfaces:
                surfs = surfs + surf.num_faces
            for domain in domains:
                voxels = voxels + domain.num_voxels

            m_file.write(b"$Elements\n")
            num_blocks = len(face_elements) + len(voxel_elements)
            min_tag = 1
            max_tag = surfs + voxels

            m_file.write("{} {} {} {}\n".format(num_blocks, surfs + voxels, min_tag, max_tag).encode("utf-8"))

            ta = 1
            for i in range(0, len(face_elements)):
                num_elem = face_elements[i].shape[0]

                m_file.write("{} {} {} {}\n".format(2, i + 1, 2, num_elem).encode("utf-8"))
                np.savetxt(m_file, np.column_stack([np.arange(ta, ta + num_elem), face_elements[i]]), "%d", " ")
                ta = ta + num_elem
                m_file.write(b"\n")

            for i in range(0, len(voxel_elements)):
                num_elem = voxel_elements[i].shape[0]

                m_file.write("{} {} {} {}\n".format(3, i + 1, 4, num_elem).encode("utf-8"))
                np.savetxt(m_file, np.column_stack([np.arange(ta, ta + num_elem), voxel_elements[i]]), "%d", " ")
                ta = ta + num_elem
                m_file.write(b"\n")
            m_file.write(b"$EndElements\n")