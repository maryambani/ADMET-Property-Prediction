# short background on each Tox21 assay for the app tooltips.
# assay panel + PubChem AIDs come from Huang et al. 2016 [1]; mechanism notes are
# summarised from the cited Tox21 papers plus standard toxicology background.

ASSAYS = {
    "NR-AR": {
        "target": "androgen receptor (full length)",
        "about": (
            "The androgen receptor is switched on by hormones like testosterone and drives "
            "male sexual development and reproductive function. Chemicals that activate or "
            "block it can act as endocrine disruptors, interfering with hormone signalling "
            "and development."
        ),
        "refs": [1],
    },
    "NR-AR-LBD": {
        "target": "androgen receptor, ligand-binding domain",
        "about": (
            "Same receptor, but this assay isolates the ligand-binding domain, the pocket "
            "where hormones dock. A hit here suggests the compound binds the receptor "
            "directly rather than acting elsewhere in the pathway."
        ),
        "refs": [1],
    },
    "NR-AhR": {
        "target": "aryl hydrocarbon receptor",
        "about": (
            "AhR is a sensor for environmental pollutants such as dioxins and polycyclic "
            "aromatic hydrocarbons, and it responds by switching on drug-metabolising enzymes "
            "(e.g. CYP1A1). Sustained activation is associated with toxicity, immune effects "
            "and carcinogenesis."
        ),
        "refs": [1],
    },
    "NR-Aromatase": {
        "target": "aromatase (CYP19A1) inhibition",
        "about": (
            "Aromatase is the enzyme that converts androgens into estrogens, so it controls "
            "the balance between the two hormone classes. Inhibiting it lowers estrogen "
            "production, which can disrupt reproduction and development."
        ),
        "refs": [1, 5],
    },
    "NR-ER": {
        "target": "estrogen receptor alpha (full length)",
        "about": (
            "Estrogen receptor alpha regulates reproductive tissues, bone and many other "
            "organs. Chemicals that mimic estrogen (such as bisphenol A) or block it are "
            "classic endocrine disruptors."
        ),
        "refs": [1, 6],
    },
    "NR-ER-LBD": {
        "target": "estrogen receptor alpha, ligand-binding domain",
        "about": (
            "Isolates the hormone-binding pocket of the estrogen receptor. Activity here "
            "points to direct binding to the receptor rather than an indirect effect on "
            "estrogen signalling."
        ),
        "refs": [1, 6],
    },
    "NR-PPAR-gamma": {
        "target": "PPAR-gamma receptor",
        "about": (
            "PPAR-gamma is a nuclear receptor that controls fat-cell formation, lipid storage "
            "and insulin sensitivity. Chemicals that activate it may disturb metabolism, and "
            "some environmental PPAR-gamma activators are studied as suspected obesogens."
        ),
        "refs": [1],
    },
    "SR-ARE": {
        "target": "Nrf2 / antioxidant response element",
        "about": (
            "The antioxidant response element is switched on by the Nrf2 pathway when cells "
            "face oxidative or electrophilic stress. A hit means the compound is generating "
            "reactive species or reactive intermediates that cells must defend against."
        ),
        "refs": [1],
    },
    "SR-ATAD5": {
        "target": "ATAD5 stabilisation (genotoxicity)",
        "about": (
            "ATAD5 protein accumulates when DNA replication stalls or DNA is damaged, so it "
            "acts as a reporter for genotoxic stress. Activity here flags compounds that may "
            "damage DNA."
        ),
        "refs": [1, 3],
    },
    "SR-HSE": {
        "target": "heat shock factor response element",
        "about": (
            "The heat shock response is triggered when proteins misfold; the cell answers by "
            "producing chaperones (heat shock proteins) to refold or clear them. Activation "
            "indicates the compound is causing proteotoxic stress."
        ),
        "refs": [1],
    },
    "SR-MMP": {
        "target": "mitochondrial membrane potential",
        "about": (
            "Mitochondria maintain an electrical gradient across their inner membrane to make "
            "ATP, the cell's energy currency. A compound that collapses this potential is "
            "damaging mitochondria, which can starve cells of energy and trigger cell death."
        ),
        "refs": [1, 2, 4],
    },
    "SR-p53": {
        "target": "p53 DNA-damage response",
        "about": (
            "p53 is a tumour-suppressor protein activated by DNA damage and other cellular "
            "stress; it halts the cell cycle for repair or triggers cell death. Activation "
            "suggests the compound damages DNA or stresses cells enough to trip this alarm."
        ),
        "refs": [1],
    },
}

SOURCES = {
    1: (
        "Huang R, Xia M, Nguyen D-T, et al. (2016). Tox21Challenge to Build Predictive "
        "Models of Nuclear Receptor and Stress Response Pathways as Mediated by Exposure to "
        "Environmental Chemicals and Drugs. Front. Environ. Sci. 3:85.",
        "https://doi.org/10.3389/fenvs.2015.00085",
    ),
    2: (
        "Attene-Ramos MS, Huang R, Sakamuru S, et al. (2013). Systematic Study of "
        "Mitochondrial Toxicity of Environmental Chemicals Using Quantitative High "
        "Throughput Screening. Chem. Res. Toxicol. 26:1323-1332.",
        "https://doi.org/10.1021/tx4001754",
    ),
    3: (
        "Fox JT, Sakamuru S, Huang R, et al. (2012). High-throughput genotoxicity assay "
        "identifies antioxidants as inducers of DNA damage response and cell death. "
        "PNAS 109(14):5423-5428.",
        "https://doi.org/10.1073/pnas.1114278109",
    ),
    4: (
        "Attene-Ramos MS, Huang R, Michael S, et al. (2015). Profiling of the Tox21 chemical "
        "collection for mitochondrial function to identify compounds that acutely decrease "
        "mitochondrial membrane potential. Environ. Health Perspect. 123:49-56.",
        "https://doi.org/10.1289/ehp.1408642",
    ),
    5: (
        "Chen S, Hsieh J-H, Huang R, et al. (2015). Cell-based high-throughput screening for "
        "aromatase inhibitors in the Tox21 10K library. Toxicol. Sci. 147:446-457.",
        "https://doi.org/10.1093/toxsci/kfv141",
    ),
    6: (
        "Huang R, Sakamuru S, Martin MT, et al. (2014). Profiling of the Tox21 10K compound "
        "library for agonists and antagonists of the estrogen receptor alpha signaling "
        "pathway. Sci. Rep. 4:5664.",
        "https://doi.org/10.1038/srep05664",
    ),
    7: (
        "Tox21 Data Challenge 2014, assay descriptions and data. NCATS / NIH.",
        "https://tripod.nih.gov/tox21/challenge/data.jsp",
    ),
}
