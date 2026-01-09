import xml.etree.ElementTree as et
from os import remove
from zipfile import ZipFile

from lotr.em.core import EmNeuron


def neuron_from_xml(path):
    xtree = et.parse(path)
    node = xtree.getroot()
    neuron = None
    for n in list(node):  # ugly loop
        if n.tag == "thing":
            neuron = EmNeuron(n)

    return neuron


def load_skeletons_dict_from_zip(path):
    segments = load_skeletons_from_zip(path)
    return {s.id: s for s in segments}


def load_skeletons_from_zip(path, with_axon_only=False):
    content_name = "annotation.xml"
    with ZipFile(path, "r") as zip_obj:
        # Extract all the contents of zip file (the "annotation.xml" file)
        # in current directory:
        zip_obj.extractall()

    xtree = et.parse(content_name)
    node = xtree.getroot()

    skel_list = []
    for n in list(node):  # ugly loop
        if n.tag == "thing":
            skeleton = EmNeuron(n)
            if with_axon_only:
                if skeleton.has_axon:
                    skel_list.append(EmNeuron(n))
            else:
                skel_list.append(EmNeuron(n))

    remove(content_name)
    return sorted(skel_list, key=lambda n: n.id)


def load_skeletons_from_xml(path, with_axon_only=False):
    xtree = et.parse(path)
    node = xtree.getroot()

    skel_list = []
    for n in list(node):  # ugly loop
        if n.tag == "thing":
            skeleton = EmNeuron(n)
            if with_axon_only:
                if skeleton.has_axon:
                    skel_list.append(EmNeuron(n))
            else:
                skel_list.append(EmNeuron(n))

    return sorted(skel_list, key=lambda n: n.id)
