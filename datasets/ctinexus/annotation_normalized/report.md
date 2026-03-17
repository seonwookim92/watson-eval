# CTINexus Annotation Normalization Report

- Source: `/mnt/d/Ontology/watson-eval-deploy/datasets/ctinexus/annotation`
- Target: `/mnt/d/Ontology/watson-eval-deploy/datasets/ctinexus/annotation_normalized`
- Files normalized: 10
- Top-level `text` removed: 10 files
- Implicit triplet refs rewritten to strings: 18
- Alias fixes applied: 2
- Extra entities added: 3
- Entity typed payloads normalized: 220
- Relation typed payloads normalized: 78
- Semantic entity corrections: 64
- Semantic relation corrections: 76

## Canonical Schema

- Top-level keys: `entities`, `explicit_triplets`, `implicit_triplets` only
- `entity_*_type` schema: `name`, `uri`, `matched`, `match_quality`, `note`
- `relation_*_type` schema: `name`, `uri`, `matched`, `match_quality`, `is_inverse`, `note`
- All `implicit_triplets` `subject`/`object` references were rewritten to plain entity-name strings

## File-by-File Changes

### 0-days-exploited-by-commercial-surveillance-vendor-in-egypt_typed.json

- Structural normalization:
  - removed top-level `text`: yes
  - normalized entity typed payloads: 23
  - normalized relation typed payloads: 2
  - implicit ref rewrites: 2
  - no alias or missing-entity repair needed
- Semantic corrections:
  - entity 'CVE-2023-41991' entity_uco_type: NON_MATCH -> NON_MATCH (remove forced UCO vulnerability proxy)
  - entity 'CVE-2023-41992' entity_uco_type: NON_MATCH -> NON_MATCH (remove forced UCO vulnerability proxy)
  - entity 'CVE-2023-41993' entity_uco_type: NON_MATCH -> NON_MATCH (remove forced UCO vulnerability proxy)
  - explicit_triplets[2] 'an in-the-wild 0-day exploit chain | targets | iPhones' relation_uco_type: object -> NON_MATCH (remove overly generic UCO object relation)
  - explicit_triplets[8] 'CVE-2023-41991 | patched in | iOS 16.7 and iOS 17.0.1' relation_uco_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[8] 'CVE-2023-41991 | patched in | iOS 16.7 and iOS 17.0.1' relation_malont_type: hasVulnerability -> NON_MATCH (remove forced relation match)
  - explicit_triplets[8] 'CVE-2023-41991 | patched in | iOS 16.7 and iOS 17.0.1' relation_stix_type: mitigates -> NON_MATCH (remove forced relation match)
  - explicit_triplets[9] 'CVE-2023-41992 | patched in | iOS 16.7 and iOS 17.0.1' relation_uco_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[9] 'CVE-2023-41992 | patched in | iOS 16.7 and iOS 17.0.1' relation_malont_type: hasVulnerability -> NON_MATCH (remove forced relation match)
  - explicit_triplets[9] 'CVE-2023-41992 | patched in | iOS 16.7 and iOS 17.0.1' relation_stix_type: mitigates -> NON_MATCH (remove forced relation match)
  - explicit_triplets[10] 'CVE-2023-41993 | patched in | iOS 16.7 and iOS 17.0.1' relation_uco_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[10] 'CVE-2023-41993 | patched in | iOS 16.7 and iOS 17.0.1' relation_malont_type: hasVulnerability -> NON_MATCH (remove forced relation match)
  - explicit_triplets[10] 'CVE-2023-41993 | patched in | iOS 16.7 and iOS 17.0.1' relation_stix_type: mitigates -> NON_MATCH (remove forced relation match)
  - implicit_triplets[0] 'iOS 16.7 and iOS 17.0.1 | patched | an in-the-wild 0-day exploit chain' relation_uco_type: NON_MATCH -> NON_MATCH (remove non-vulnerability patched proxy)
  - implicit_triplets[0] 'iOS 16.7 and iOS 17.0.1 | patched | an in-the-wild 0-day exploit chain' relation_malont_type: NON_MATCH -> NON_MATCH (remove non-vulnerability patched proxy)
  - implicit_triplets[0] 'iOS 16.7 and iOS 17.0.1 | patched | an in-the-wild 0-day exploit chain' relation_stix_type: mitigates -> NON_MATCH (remove non-vulnerability patched proxy)
- Validation:
  - missing top-level keys: none
  - extra top-level keys: none
  - missing entity refs: 0
  - non-string triplet roles: 0
  - entity type schema issues: 0
  - relation type schema issues: 0

### 3am-ransomware-lockbit_typed.json

- Structural normalization:
  - removed top-level `text`: yes
  - normalized entity typed payloads: 14
  - normalized relation typed payloads: 0
  - implicit ref rewrites: 0
  - no alias or missing-entity repair needed
- Semantic corrections:
  - entity 'a ransomware affiliate' entity_uco_type: Person -> MaliciousRole (standardize attacker semantics in UCO)
  - entity 'a ransomware affiliate' entity_malont_type: ThreatActor -> ThreatActor (standardize attacker label in MALOnt)
  - entity 'a ransomware affiliate' entity_stix_type: Threat Actor -> Threat Actor (standardize attacker label in STIX)
  - explicit_triplets[0] '3AM | is | ransomware' relation_uco_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[0] '3AM | is | ransomware' relation_malont_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[0] '3AM | is | ransomware' relation_stix_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[1] 'Symantec's Threat Hunter Team | observed | a ransomware affiliate' relation_stix_type: investigates -> NON_MATCH (remove observed-to-investigates overmatch)
- Validation:
  - missing top-level keys: none
  - extra top-level keys: none
  - missing entity refs: 0
  - non-string triplet roles: 0
  - entity type schema issues: 0
  - relation type schema issues: 0

### active-north-korean-campaign-targeting-security-researchers_typed.json

- Structural normalization:
  - removed top-level `text`: yes
  - normalized entity typed payloads: 12
  - normalized relation typed payloads: 2
  - implicit ref rewrites: 0
  - alias fix: explicit_triplets[8].subject: 'campaign' -> 'a new campaign'
- Semantic corrections:
  - entity 'Threat Actor' entity_uco_type: Person -> MaliciousRole (standardize attacker semantics in UCO)
  - entity 'Threat Actor' entity_malont_type: ThreatActor -> ThreatActor (standardize attacker label in MALOnt)
  - entity 'Threat Actor' entity_stix_type: Threat Actor -> Threat Actor (standardize attacker label in STIX)
  - entity 'vulnerability research and development' entity_malont_type: Vulnerability -> NON_MATCH (remove activity-to-vulnerability overmatch)
  - entity 'vulnerability research and development' entity_stix_type: Vulnerability -> NON_MATCH (remove activity-to-vulnerability overmatch)
  - explicit_triplets[3] 'security researchers | worked on | vulnerability research and development' relation_stix_type: investigates -> NON_MATCH (remove broad investigates mapping)
- Validation:
  - missing top-level keys: none
  - extra top-level keys: none
  - missing entity refs: 0
  - non-string triplet roles: 0
  - entity type schema issues: 0
  - relation type schema issues: 0

### agonizing-serpens-targets-israeli-tech-higher-ed-sectors_typed.json

- Structural normalization:
  - removed top-level `text`: yes
  - normalized entity typed payloads: 10
  - normalized relation typed payloads: 3
  - implicit ref rewrites: 2
  - no alias or missing-entity repair needed
- Semantic corrections:
  - entity 'attackers' entity_uco_type: Person -> MaliciousRole (standardize attacker semantics in UCO)
  - entity 'attackers' entity_malont_type: ThreatActor -> ThreatActor (standardize attacker label in MALOnt)
  - entity 'attackers' entity_stix_type: Threat Actor -> Threat Actor (standardize attacker label in STIX)
  - entity 'sensitive data' entity_stix_type: Observed Data -> NON_MATCH (remove abstract-information to Observed Data overmatch)
  - entity 'personally identifiable information (PII)' entity_stix_type: Observed Data -> NON_MATCH (remove abstract-information to Observed Data overmatch)
  - entity 'intellectual property' entity_stix_type: Observed Data -> NON_MATCH (remove abstract-information to Observed Data overmatch)
  - explicit_triplets[4] 'attackers | attempted to steal | sensitive data' relation_malont_type: targets -> NON_MATCH (remove broad targets mapping)
  - explicit_triplets[4] 'attackers | attempted to steal | sensitive data' relation_stix_type: compromises -> NON_MATCH (remove broad compromises mapping)
  - explicit_triplets[8] 'wipers | intended to achieve | cover the attackers' tracks' relation_malont_type: uses -> NON_MATCH (remove outcome-to-uses overmatch)
  - explicit_triplets[8] 'wipers | intended to achieve | cover the attackers' tracks' relation_stix_type: uses -> NON_MATCH (remove outcome-to-uses overmatch)
  - explicit_triplets[9] 'wipers | intended to achieve | render the infected endpoints unusable' relation_malont_type: uses -> NON_MATCH (remove outcome-to-uses overmatch)
  - explicit_triplets[9] 'wipers | intended to achieve | render the infected endpoints unusable' relation_stix_type: uses -> NON_MATCH (remove outcome-to-uses overmatch)
- Validation:
  - missing top-level keys: none
  - extra top-level keys: none
  - missing entity refs: 0
  - non-string triplet roles: 0
  - entity type schema issues: 0
  - relation type schema issues: 0

### amd-apple-qualcomm-gpus-leak-ai-data-in-leftoverlocals-attacks_typed.json

- Structural normalization:
  - removed top-level `text`: yes
  - normalized entity typed payloads: 24
  - normalized relation typed payloads: 2
  - implicit ref rewrites: 0
  - extra entity: added entity 'uninitialized local memory' cloned from 'local memory' with entity_id=13
- Semantic corrections:
  - entity 'LeftoverLocals' entity_uco_type: NON_MATCH -> NON_MATCH (remove forced UCO vulnerability proxy)
  - entity 'CVE-2023-4969' entity_uco_type: NON_MATCH -> NON_MATCH (remove forced UCO vulnerability proxy)
  - entity 'attacker' entity_uco_type: Person -> MaliciousRole (standardize attacker semantics in UCO)
  - entity 'attacker' entity_malont_type: ThreatActor -> ThreatActor (standardize attacker label in MALOnt)
  - entity 'attacker' entity_stix_type: Threat Actor -> Threat Actor (standardize attacker label in STIX)
  - explicit_triplets[0] 'LeftoverLocals | affects | graphics processing units' relation_uco_type: NON_MATCH -> NON_MATCH (keep conservative UCO non-match)
  - explicit_triplets[0] 'LeftoverLocals | affects | graphics processing units' relation_malont_type: targets -> hasVulnerability (prefer inverse hasVulnerability over generic targets)
  - explicit_triplets[0] 'LeftoverLocals | affects | graphics processing units' relation_stix_type: targets -> NON_MATCH (remove forced STIX vulnerability-affects proxy)
  - explicit_triplets[9] 'a listener | dumps data into | global memory' relation_stix_type: drops -> NON_MATCH (remove drops overmatch)
- Validation:
  - missing top-level keys: none
  - extra top-level keys: none
  - missing entity refs: 0
  - non-string triplet roles: 0
  - entity type schema issues: 0
  - relation type schema issues: 0

### an-in-depth-look-at-cuba-ransomware_typed.json

- Structural normalization:
  - removed top-level `text`: yes
  - normalized entity typed payloads: 18
  - normalized relation typed payloads: 2
  - implicit ref rewrites: 2
  - alias fix: explicit_triplets[7].subject: 'Cuba ransomware operators' -> 'Cuba ransomware actors'
  - extra entity: added entity 'Industrial Spy ransomware actors' cloned from 'Cuba ransomware actors' with entity_id=15
- Semantic corrections:
  - entity '$145 million' entity_stix_type: Observed Data -> NON_MATCH (remove abstract-information to Observed Data overmatch)
  - entity '$145 million' entity_uco_type: ContentData -> NON_MATCH (remove monetary-value overmatch)
  - entity '$60 million' entity_stix_type: Observed Data -> NON_MATCH (remove abstract-information to Observed Data overmatch)
  - entity '$60 million' entity_uco_type: ContentData -> NON_MATCH (remove monetary-value overmatch)
  - entity 'Cuba ransomware actors' entity_uco_type: Person -> MaliciousRole (standardize attacker semantics in UCO)
  - entity 'Cuba ransomware actors' entity_malont_type: ThreatActor -> ThreatActor (standardize attacker label in MALOnt)
  - entity 'Cuba ransomware actors' entity_stix_type: Threat Actor -> Threat Actor (standardize attacker label in STIX)
  - entity 'RomCom Remote Access Trojan (RAT) actors' entity_uco_type: Person -> MaliciousRole (standardize attacker semantics in UCO)
  - entity 'RomCom Remote Access Trojan (RAT) actors' entity_malont_type: ThreatActor -> ThreatActor (standardize attacker label in MALOnt)
  - entity 'RomCom Remote Access Trojan (RAT) actors' entity_stix_type: Threat Actor -> Threat Actor (standardize attacker label in STIX)
  - entity 'Industrial Spy ransomware actors' entity_uco_type: Person -> MaliciousRole (standardize attacker semantics in UCO)
  - entity 'Industrial Spy ransomware actors' entity_malont_type: ThreatActor -> ThreatActor (standardize attacker label in MALOnt)
  - entity 'Industrial Spy ransomware actors' entity_stix_type: Threat Actor -> Threat Actor (standardize attacker label in STIX)
  - explicit_triplets[12] 'Cuba ransomware | took responsibility for | a cyberattack on The Philadelphia Inquirer' relation_stix_type: attributed-to -> NON_MATCH (remove inverse attribution overmatch)
- Validation:
  - missing top-level keys: none
  - extra top-level keys: none
  - missing entity refs: 0
  - non-string triplet roles: 0
  - entity type schema issues: 0
  - relation type schema issues: 0

### an-in-depth-look-at-quantum-ransomware_typed.json

- Structural normalization:
  - removed top-level `text`: yes
  - normalized entity typed payloads: 31
  - normalized relation typed payloads: 8
  - implicit ref rewrites: 2
  - extra entity: added entity 'Conti Team Two' cloned from 'Conti' with entity_id=27
- Semantic corrections:
  - entity 'Quantum' entity_uco_type: Organization -> MaliciousRole (standardize attacker semantics in UCO)
  - entity 'Quantum' entity_malont_type: ThreatActor -> ThreatActor (standardize attacker label in MALOnt)
  - entity 'Quantum' entity_stix_type: ThreatActor -> Threat Actor (standardize attacker label in STIX)
  - entity 'Conti' entity_uco_type: Organization -> MaliciousRole (standardize attacker semantics in UCO)
  - entity 'Conti' entity_malont_type: ThreatActor -> ThreatActor (standardize attacker label in MALOnt)
  - entity 'Conti' entity_stix_type: ThreatActor -> Threat Actor (standardize attacker label in STIX)
  - entity 'Conti Team Two' entity_uco_type: Organization -> MaliciousRole (standardize attacker semantics in UCO)
  - entity 'Conti Team Two' entity_malont_type: ThreatActor -> ThreatActor (standardize attacker label in MALOnt)
  - entity 'Conti Team Two' entity_stix_type: ThreatActor -> Threat Actor (standardize attacker label in STIX)
  - implicit_triplets[0] 'Quantum | is linked to | Quantum Locker' relation_malont_type: uses -> NON_MATCH (remove vague-link to uses overmatch)
  - implicit_triplets[0] 'Quantum | is linked to | Quantum Locker' relation_stix_type: uses -> NON_MATCH (remove vague-link to uses overmatch)
- Validation:
  - missing top-level keys: none
  - extra top-level keys: none
  - missing entity refs: 0
  - non-string triplet roles: 0
  - entity type schema issues: 0
  - relation type schema issues: 0

### an-in-depth-look-at-rhysida-ransomware_typed.json

- Structural normalization:
  - removed top-level `text`: yes
  - normalized entity typed payloads: 31
  - normalized relation typed payloads: 8
  - implicit ref rewrites: 4
  - no alias or missing-entity repair needed
- Semantic corrections:
  - entity 'active directory (AD) password' entity_stix_type: User Account -> NON_MATCH (remove password-to-user-account overmatch)
  - entity 'Threat Actor' entity_uco_type: Person -> MaliciousRole (standardize attacker semantics in UCO)
  - entity 'Threat Actor' entity_malont_type: ThreatActor -> ThreatActor (standardize attacker label in MALOnt)
  - entity 'Threat Actor' entity_stix_type: Threat Actor -> Threat Actor (standardize attacker label in STIX)
  - entity 'AES-CTR' entity_stix_type: EncryptionAlgorithmEnum -> NON_MATCH (remove enum-class overmatch)
- Validation:
  - missing top-level keys: none
  - extra top-level keys: none
  - missing entity refs: 0
  - non-string triplet roles: 0
  - entity type schema issues: 0
  - relation type schema issues: 0

### apache-erp-0day-underscores-dangers-of-incomplete-patches_typed.json

- Structural normalization:
  - removed top-level `text`: yes
  - normalized entity typed payloads: 33
  - normalized relation typed payloads: 30
  - implicit ref rewrites: 4
  - no alias or missing-entity repair needed
- Semantic corrections:
  - entity 'Unknown groups' entity_uco_type: Identity -> MaliciousRole (standardize attacker semantics in UCO)
  - entity 'Unknown groups' entity_malont_type: ThreatActor -> ThreatActor (standardize attacker label in MALOnt)
  - entity 'Unknown groups' entity_stix_type: Threat Actor -> Threat Actor (standardize attacker label in STIX)
  - entity 'zero-day vulnerability CVE-2023-51467' entity_uco_type: NON_MATCH -> NON_MATCH (remove forced UCO vulnerability proxy)
  - entity 'access sensitive information' entity_uco_type: Action -> ActionPattern (prefer ActionPattern over generic Action)
  - entity 'remotely execute code against applications' entity_uco_type: Action -> ActionPattern (prefer ActionPattern over generic Action)
  - entity 'CVE-2023-49070' entity_uco_type: NON_MATCH -> NON_MATCH (remove forced UCO vulnerability proxy)
  - explicit_triplets[0] 'Unknown groups | launched probes against | zero-day vulnerability CVE-2023-51467' relation_uco_type: target -> NON_MATCH (remove forced relation match)
  - explicit_triplets[0] 'Unknown groups | launched probes against | zero-day vulnerability CVE-2023-51467' relation_malont_type: targets -> NON_MATCH (remove forced relation match)
  - explicit_triplets[0] 'Unknown groups | launched probes against | zero-day vulnerability CVE-2023-51467' relation_stix_type: targets -> NON_MATCH (remove forced relation match)
  - explicit_triplets[1] 'zero-day vulnerability CVE-2023-51467 | affects | Apache OfBiz' relation_uco_type: NON_MATCH -> NON_MATCH (keep conservative UCO non-match)
  - explicit_triplets[1] 'zero-day vulnerability CVE-2023-51467 | affects | Apache OfBiz' relation_malont_type: hasVulnerability -> hasVulnerability (prefer inverse hasVulnerability over generic targets)
  - explicit_triplets[1] 'zero-day vulnerability CVE-2023-51467 | affects | Apache OfBiz' relation_stix_type: NON_MATCH -> NON_MATCH (remove forced STIX vulnerability-affects proxy)
  - explicit_triplets[2] 'zero-day vulnerability CVE-2023-51467 | allows attacker to | access sensitive information' relation_uco_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[2] 'zero-day vulnerability CVE-2023-51467 | allows attacker to | access sensitive information' relation_malont_type: targets -> NON_MATCH (remove forced relation match)
  - explicit_triplets[2] 'zero-day vulnerability CVE-2023-51467 | allows attacker to | access sensitive information' relation_stix_type: exploits -> NON_MATCH (remove forced relation match)
  - explicit_triplets[3] 'zero-day vulnerability CVE-2023-51467 | allows attacker to | remotely execute code against applications' relation_uco_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[3] 'zero-day vulnerability CVE-2023-51467 | allows attacker to | remotely execute code against applications' relation_malont_type: targets -> NON_MATCH (remove forced relation match)
  - explicit_triplets[3] 'zero-day vulnerability CVE-2023-51467 | allows attacker to | remotely execute code against applications' relation_stix_type: exploits -> NON_MATCH (remove forced relation match)
  - explicit_triplets[5] 'patch for CVE-2023-49070 | failed to protect against | other variations of the attack' relation_uco_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[5] 'patch for CVE-2023-49070 | failed to protect against | other variations of the attack' relation_malont_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[5] 'patch for CVE-2023-49070 | failed to protect against | other variations of the attack' relation_stix_type: mitigates -> NON_MATCH (remove forced relation match)
  - explicit_triplets[6] 'zero-day vulnerability CVE-2023-51467 | disclosed on | Dec. 26' relation_uco_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[6] 'zero-day vulnerability CVE-2023-51467 | disclosed on | Dec. 26' relation_malont_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[6] 'zero-day vulnerability CVE-2023-51467 | disclosed on | Dec. 26' relation_stix_type: published -> NON_MATCH (remove forced relation match)
  - explicit_triplets[7] 'SonicWall | analyzed | zero-day vulnerability CVE-2023-51467' relation_uco_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[7] 'SonicWall | analyzed | zero-day vulnerability CVE-2023-51467' relation_malont_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[7] 'SonicWall | analyzed | zero-day vulnerability CVE-2023-51467' relation_stix_type: analysis-of -> NON_MATCH (remove forced relation match)
  - implicit_triplets[1] 'patch for CVE-2023-49070 | failed to protect against | zero-day vulnerability CVE-2023-51467' relation_uco_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - implicit_triplets[1] 'patch for CVE-2023-49070 | failed to protect against | zero-day vulnerability CVE-2023-51467' relation_malont_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - implicit_triplets[1] 'patch for CVE-2023-49070 | failed to protect against | zero-day vulnerability CVE-2023-51467' relation_stix_type: mitigates -> NON_MATCH (remove forced relation match)
- Validation:
  - missing top-level keys: none
  - extra top-level keys: none
  - missing entity refs: 0
  - non-string triplet roles: 0
  - entity type schema issues: 0
  - relation type schema issues: 0

### apple-fixes-first-zero-day-bug-exploited-in-attacks-this-year_typed.json

- Structural normalization:
  - removed top-level `text`: yes
  - normalized entity typed payloads: 24
  - normalized relation typed payloads: 21
  - implicit ref rewrites: 2
  - no alias or missing-entity repair needed
- Semantic corrections:
  - entity 'zero-day vulnerability CVE-2024-23222' entity_uco_type: Software -> NON_MATCH (remove forced UCO vulnerability proxy)
  - entity 'WebKit confusion issue' entity_uco_type: Code -> NON_MATCH (remove UCO Code proxy for vulnerability description)
  - entity 'attackers' entity_uco_type: MaliciousRole -> MaliciousRole (standardize attacker semantics in UCO)
  - entity 'attackers' entity_malont_type: ThreatActor -> ThreatActor (standardize attacker label in MALOnt)
  - entity 'attackers' entity_stix_type: Threat Actor -> Threat Actor (standardize attacker label in STIX)
  - entity 'threat actors' entity_uco_type: MaliciousRole -> MaliciousRole (standardize attacker semantics in UCO)
  - entity 'threat actors' entity_malont_type: ThreatActor -> ThreatActor (standardize attacker label in MALOnt)
  - entity 'threat actors' entity_stix_type: Threat Actor -> Threat Actor (standardize attacker label in STIX)
  - explicit_triplets[0] 'Apple | released security updates to address | zero-day vulnerability CVE-2024-23222' relation_uco_type: NON_MATCH -> NON_MATCH (normalize generic UCO wrapper to non-match)
  - explicit_triplets[0] 'Apple | released security updates to address | zero-day vulnerability CVE-2024-23222' relation_malont_type: NON_MATCH -> NON_MATCH (keep MALOnt non-match)
  - explicit_triplets[0] 'Apple | released security updates to address | zero-day vulnerability CVE-2024-23222' relation_stix_type: mitigates -> mitigates (preserve valid STIX remediation relation)
  - explicit_triplets[1] 'zero-day vulnerability CVE-2024-23222 | impacts | Operating System' relation_uco_type: NON_MATCH -> NON_MATCH (keep conservative UCO non-match)
  - explicit_triplets[1] 'zero-day vulnerability CVE-2024-23222 | impacts | Operating System' relation_malont_type: hasVulnerability -> hasVulnerability (prefer inverse hasVulnerability over generic targets)
  - explicit_triplets[1] 'zero-day vulnerability CVE-2024-23222 | impacts | Operating System' relation_stix_type: NON_MATCH -> NON_MATCH (remove forced STIX vulnerability-affects proxy)
  - explicit_triplets[2] 'zero-day vulnerability CVE-2024-23222 | is | WebKit confusion issue' relation_uco_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[2] 'zero-day vulnerability CVE-2024-23222 | is | WebKit confusion issue' relation_malont_type: has -> NON_MATCH (remove forced relation match)
  - explicit_triplets[2] 'zero-day vulnerability CVE-2024-23222 | is | WebKit confusion issue' relation_stix_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[3] 'attackers | could exploit | zero-day vulnerability CVE-2024-23222' relation_uco_type: NON_MATCH -> NON_MATCH (normalize generic UCO wrapper to non-match)
  - explicit_triplets[3] 'attackers | could exploit | zero-day vulnerability CVE-2024-23222' relation_malont_type: targets -> NON_MATCH (remove broad targets mapping)
  - explicit_triplets[3] 'attackers | could exploit | zero-day vulnerability CVE-2024-23222' relation_stix_type: exploits -> exploits (preserve direct STIX exploit relation)
  - explicit_triplets[4] 'threat actors | could execute | arbitrary malicious code' relation_uco_type: NON_MATCH -> NON_MATCH (normalize generic UCO wrapper to non-match)
  - explicit_triplets[4] 'threat actors | could execute | arbitrary malicious code' relation_malont_type: uses -> uses (preserve approximate MALOnt uses relation)
  - explicit_triplets[4] 'threat actors | could execute | arbitrary malicious code' relation_stix_type: uses -> uses (preserve direct STIX uses relation)
  - explicit_triplets[5] 'arbitrary malicious code | runs on | devices running vulnerable iOS, macOS, and tvOS versions' relation_uco_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - explicit_triplets[5] 'arbitrary malicious code | runs on | devices running vulnerable iOS, macOS, and tvOS versions' relation_malont_type: uses -> NON_MATCH (remove forced relation match)
  - explicit_triplets[5] 'arbitrary malicious code | runs on | devices running vulnerable iOS, macOS, and tvOS versions' relation_stix_type: hosts -> NON_MATCH (remove forced relation match)
  - implicit_triplets[0] 'zero-day vulnerability CVE-2024-23222 | enables execution of | arbitrary malicious code' relation_uco_type: NON_MATCH -> NON_MATCH (remove forced relation match)
  - implicit_triplets[0] 'zero-day vulnerability CVE-2024-23222 | enables execution of | arbitrary malicious code' relation_malont_type: exploits -> NON_MATCH (remove forced relation match)
  - implicit_triplets[0] 'zero-day vulnerability CVE-2024-23222 | enables execution of | arbitrary malicious code' relation_stix_type: exploits -> NON_MATCH (remove forced relation match)
- Validation:
  - missing top-level keys: none
  - extra top-level keys: none
  - missing entity refs: 0
  - non-string triplet roles: 0
  - entity type schema issues: 0
  - relation type schema issues: 0

