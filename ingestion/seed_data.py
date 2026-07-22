from typing import List, Dict, Any

SAMPLE_SCIENTIFIC_KNOWLEDGE: List[Dict[str, Any]] = [
    {
        "id": "doc_alz_01",
        "title": "Amyloid-Beta 42 and Tau Protein Aggregation Kinetics in Early Alzheimer's Disease",
        "content": "Aβ42 soluble oligomers induce microglial activation and oxidative stress pathways in cortical neurons. Hyperphosphorylated Tau accumulation leads to microtubule disassembly and axonal transport disruption. Key biomolecules: Aβ42, Tau, APOE4.",
        "domain": "neuroscience",
        "year": 2023,
        "authors": ["Dr. S. Chen", "Dr. A. Miller"],
        "allowed_roles": ["public", "researcher"],
        "tags": ["Alzheimer's", "Aβ42", "Tau", "neurodegeneration", "biomarkers"]
    },
    {
        "id": "doc_alz_02",
        "title": "APOE4 Allele Regulation of Neuroinflammation and Blood-Brain Barrier Leakage",
        "content": "Apolipoprotein E4 (APOE4) compromises blood-brain barrier (BBB) integrity via the CypA-NFkB pathway in pericytes, impairing clearance of toxic amyloid proteins and exacerbating central neuroinflammation.",
        "domain": "neuroscience",
        "year": 2022,
        "authors": ["Dr. R. Zlokovic", "Dr. B. Zhao"],
        "allowed_roles": ["public", "researcher"],
        "tags": ["APOE4", "BBB", "neuroinflammation", "biomarkers"]
    },
    {
        "id": "doc_nano_01",
        "title": "Surface-Functionalized Lipid Nanoparticles for Target-Specific Drug Delivery Across the Blood-Brain Barrier",
        "content": "Lipid nanoparticles (LNPs) functionalized with transferrin receptor-targeting antibodies or ApoE peptides exhibit high BBB transcytosis (>12% injected dose/g tissue). LNPs effectively encapsulate small molecules and mRNA cargo.",
        "domain": "nanotechnology",
        "year": 2024,
        "authors": ["Dr. M. Garcia", "Dr. K. Patel"],
        "allowed_roles": ["public", "researcher"],
        "tags": ["lipid nanoparticles", "LNP", "nanomaterials", "drug delivery", "BBB crossing"]
    },
    {
        "id": "doc_nano_02",
        "title": "Polymeric Nanoparticles and Dendrimers as Amyloid Fibrillation Inhibitors",
        "content": "Polyamidoamine (PAMAM) dendrimers and PEGylated poly(lactic-co-glycolic acid) (PLGA) nanoparticles interact with hydrophobic domains of Aβ monomers, disrupting beta-sheet aggregation and neutralising oligomer neurotoxicity.",
        "domain": "nanotechnology",
        "year": 2023,
        "authors": ["Dr. J. Wang", "Dr. L. Kumar"],
        "allowed_roles": ["public", "researcher"],
        "tags": ["dendrimers", "PLGA nanoparticles", "Aβ42 aggregation", "inhibition", "nanomaterials"]
    },
    {
        "id": "doc_cross_01",
        "title": "Biocompatible Nanomaterial Conjugates for In Vivo Neurodegenerative Biomarker Imaging and Targeted Therapy",
        "content": "Engineered gold nanoparticle clusters bioconjugated with anti-Tau monoclonal antibodies allow dual PET-MRI imaging of fibrillar Tau deposits in living cortex while enabling controlled pulse releasing of neuroprotective drug payloads.",
        "domain": "cross_domain",
        "year": 2024,
        "authors": ["Dr. H. Tanaka", "Dr. E. Rossi"],
        "allowed_roles": ["public", "researcher"],
        "tags": ["nanomaterials", "Tau", "imaging", "targeted therapy", "cross-domain"]
    },
    {
        "id": "doc_cross_02",
        "title": "Exosome-Mimetic Biomimetic Nanocarriers for MicroRNA Delivery in Neurodegenerative Pathologies",
        "content": "Biomimetic nanocarriers derived from brain endothelial exosomes encapsulate miR-124 to downregulate microglial BACE1 and inflammatory cytokine secretion, showing low cytotoxicity and high biocompatibility.",
        "domain": "cross_domain",
        "year": 2024,
        "authors": ["Dr. V. Fernandez", "Dr. T. Schmidt"],
        "allowed_roles": ["public", "researcher"],
        "tags": ["nanocarriers", "biocompatibility", "exosomes", "BACE1", "microRNA"]
    }
]

def seed_default_knowledge():
    from ingestion.pipeline import get_ingestion_pipeline
    pipeline = get_ingestion_pipeline()
    pipeline.ingest_documents(SAMPLE_SCIENTIFIC_KNOWLEDGE)
