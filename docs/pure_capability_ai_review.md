# AI-assisted review of selected PURE capabilities

Reviewed all 180 selected PURE seeds against their original requirements and document context on 2026-09-11. This is an AI-assisted source-content review, not human validation, inter-rater agreement, or an assessment of model outputs.

Decisions: 129 accept and 51 revise. EIRENE contributes 85 accepts and 27 revisions; ERTMS contributes 44 accepts and 24 revisions. Every selected seed ID appears exactly once in the decision tables, and every revision includes a complete replacement clause. No selected item lacks a disposition.

The reviewed proposals are the 180 rows in `docs/pure_capability_revisions.csv`. They matched `outputs/pure_capability_revision_review_candidate.csv` at review time. The candidate's source text, source corpus, requirement identifiers, section paths, and adjacent context matched `data/processed/seeds_review_pure.csv` for every selected row. All 180 selected rows had `include=yes`. The selection comprises 112 EIRENE requirements and 68 ERTMS requirements.

Each decision below concerns the proposed clause after “The system must/should/may” or “It would be useful if the system could”. `accept` retains the reviewed proposal. `revise` supplies its complete replacement capability in the final column. The review preserves the capability, participants, conditions, quantities, negative constraints, and relevant distinctions while removing the original requirement-level obligation. Permission and prerequisites that are part of system behavior remain in the capability. A generic system subject is interpreted as the relevant railway system or its named component. This is capability abstraction, not a claim that assigning an obligation to a system is identical to the original allocation of responsibilities to operators or designers.

Context was used to resolve procedural references and restrictions, rather than treating a detached sentence as an unconditional requirement. The archived XML in `data/raw/pure_requirements_xml.zip`, entry `XMLZIPFile/2007-eirene_fun_7-2.xml`, was additionally inspected for requirements 5.2.3.9, 5.2.3.26, 5.2.3.46, and 9.4.2. In particular, 5.2.3.9 contains four housekeeping list items that are absent from the flattened neighboring-source sentence. Other expanded references below are supported by the supplied section and neighboring records or the same source document's nearby records in the selected-source CSV.

## Row-level decisions

### EIRENE Functional Requirements Specification

| Seed ID | Source requirement | Decision | Rationale | Corrected capability if revised |
| --- | --- | --- | --- | --- |
| S0003 | 2.2.2 | accept | Retains all voice services and every fixed/mobile user combination. | — |
| S0004 | 2.2.3 | accept | Retains point-to-point voice calls between any two parties. | — |
| S0005 | 2.2.4 | accept | Resolves “such calls” and preserves simultaneous speech by both parties. | — |
| S0008 | 2.2.7 | accept | Directly preserves broadcast voice-call support. | — |
| S0020 | 2.3.2 | accept | Preserves both transmission patterns and the ground-to-mobile direction. | — |
| S0021 | 2.3.3 | accept | Preserves network support and mobile-to-ground receipt. | — |
| S0024 | 2.3.6 | accept | Preserves network support for point-to-point data communication. | — |
| S0029 | 2.3.11 | accept | Preserves fax transmission between ground and mobile users. | — |
| S0040 | 2.4.6 | accept | “Override” preserves the source's pre-emption action and higher/lower priority relation. | — |
| S0045 | 2.4.11 | accept | Preserves the forwarding user, voice-call condition, recipient, and prior conversation. | — |
| S0047 | 2.4.13 | accept | Preserves temporary exit through putting the existing call on hold. | — |
| S0049 | 2.4.15 | accept | Preserves notifying an already-engaged user of other users' contact attempts. | — |
| S0050 | 2.4.16 | accept | Preserves chargeability as the condition and both rates and ongoing charges. | — |
| S0057 | 3.2.3 | accept | Preserves support for all compliant mobiles. | — |
| S0088 | 4.3.5 | revise | Main-agent reconciliation: describe supplied equipment rather than assigning design assurance to the runtime system; retain all activities and ISO 9001. | provide EIRENE mobile radio equipment whose design, testing and installation all comply with the quality procedures defined in ISO 9001 |
| S0096 | 5.1.2 | accept | Preserves provision of a handheld portable for the driver outside the train. | — |
| S0108 | 5.2.2.4 | revise | Identify the controller call from 5.2.2.3–5 so the indication is not detached from its call type. | provide the driver with an audible and visual indication that the call to the controller is proceeding |
| S0111 | 5.2.2.7 | revise | Make the description's availability condition and driver display recipient explicit from 5.2.2.6–7. | display the connected party's alphanumeric functional-identity description to the driver when that description is present |
| S0115 | 5.2.2.11 | accept | Preserves group identity, Cab-radio displays, and participating drivers. | — |
| S0129 | 5.2.2.28 | revise | Resolve the ongoing call as the multi-driver call in 5.2.2.26–28. | display a multi-driver indication permanently at all Cab radios while the multi-driver call is ongoing |
| S0131 | 5.2.2.30 | revise | Resolve the group/call as the multi-driver call; preserve lead-driver authority and any-time access. | allow the lead driver to remove a member from the multi-driver call at any time during that call |
| S0133 | 5.2.2.32 | accept | Preserves the disconnection condition and clear indication for the multi-driver call. | — |
| S0141 | 5.2.2.40 | revise | “At that stage” remains unresolved; 5.2.2.39 identifies presentation of the staff list. | prompt the driver to select the train staff with whom to communicate after presenting the list of generic train staff |
| S0151 | 5.2.2.50 | accept | Preserves the ongoing broadcast condition, driver, visual indication, and MMI. | — |
| S0154 | 5.2.2.52 | accept | Preserves the group-call condition and driver reminder to use PTT. | — |
| S0168 | 5.2.2.66 | accept | Preserves the link-assurance tone and loudspeaker output. | — |
| S0171 | 5.2.2.69 | accept | Retains both mode transitions and the implementation condition. | — |
| S0180 | 5.2.3.3 | accept | Preserves power-down as the trigger for network disconnection. | — |
| S0181 | 5.2.3.4 | revise | Clarify that every number stored at switch-off remains retained while off. | retain all numbers stored in the Cab radio at switch-off while the radio is switched off |
| S0187 | 5.2.3.10 | revise | Resolve “above procedure” using all four XML housekeeping items; retain the power-failure condition and feasibility qualification. | complete Cab radio soft-switch-off housekeeping on power failure as far as possible, including controlled termination of a current call, train-number deregistration where applicable, required-data storage and Railway emergency-call confirmation |
| S0213 | 5.2.3.28 | revise | Replace “option 2” with its meaning and the leading-driver restriction stated in 5.2.3.26. | allow the leading driver to accept or reject the train number returned by the network after an automated train-number request |
| S0216 | 5.2.3.29 | accept | Preserves both warning channels and duplicate-number registration on the same network. | — |
| S0220 | 5.2.3.33 | revise | Restore the system as recipient of the driver's location information. | allow non-leading drivers to indicate their location in the train to the system during registration, such as second or third driver |
| S0221 | 5.2.3.34 | revise | Make the registration/deregistration correspondence explicit; the two “or” clauses obscure the matched transitions. | automatically register the driver's functional number when the train number is registered and deregister it when the train number is deregistered |
| S0233 | 5.2.3.46 | revise | Identify radio/MMI test results and treat power-on as identifying the comparison self-test, not a new test-display trigger. | display the results of radio and MMI tests in a format similar to the self-test results displayed when the MMI is powered on |
| S0238 | 5.2.4.5 | accept | Preserves holding versus clearing established calls to satisfy other priorities. | — |
| S0241 | 5.2.4.8 | accept | Preserves the pre-emption condition and advisory indication to affected parties. | — |
| S0243 | 5.2.4.10 | accept | Preserves Cab-radio diagnostic testing, all physical interfaces, and the driver's request. | — |
| S0245 | 5.2.4.12 | accept | Preserves all failures, recording availability rather than compulsory recording, and the implementation condition. | — |
| S0259 | 5.4.5 | accept | Preserves driver adjustment of buttons, indicators, and displays for ambient lighting. | — |
| S0260 | 5.4.6 | accept | Preserves the driver's display-contrast adjustment. | — |
| S0262 | 5.4.8 | accept | Preserves clear readability from normal driver position and reading distance. | — |
| S0269 | 5.4.16 | accept | Preserves loss of network contact as the trigger and both Cab-radio indication channels. | — |
| S0270 | 5.7.1 | accept | Preserves the interface, DSD-equipped traction-unit condition, and alarm-transmission purpose. | — |
| S0272 | 5.7.2 | accept | Preserves automatic Cab-radio data transmission on DSD activation. | — |
| S0274 | 5.7.4 | accept | Preserves DSD alarm information and the primary controller destination. | — |
| S0275 | 5.7.5 | revise | Resolve “additional information” as additional DSD alarm information from 5.7.3–4. | provide additional DSD alarm information if it is available from external systems |
| S0277 | 5.9.1 | accept | Preserves on-train mobiles as interface hosts and ERTMS/ETCS as the endpoint. | — |
| S0279 | 5.10.1 | accept | Preserves additional Cab-radio interfaces; their types are intentionally unspecified in the source. | — |
| S0287 | 6.2.2.6 | accept | Preserves calling-party identity and availability condition. | — |
| S0288 | 6.2.2.7 | accept | Preserves short duration and both incoming-call indication channels. | — |
| S0291 | 6.2.2.11 | accept | Preserves the group-call condition and visual reminder to use PTT. | — |
| S0303 | 6.2.3.7 | accept | Preserves first power-on and the national railway's default display language. | — |
| S0305 | 6.2.3.9 | accept | Preserves automatic network change as the trigger and both indication channels. | — |
| S0308 | 6.2.3.11 | accept | Preserves user initiation conditional on a functional-number change being required. | — |
| S0311 | 6.2.3.14 | accept | Preserves a standard data interface and its computer-to-radio connection purpose. | — |
| S0318 | 6.3.7 | accept | Preserves General purpose radio compatibility with a car adapter kit. | — |
| S0320 | 6.4.1.2 | accept | Resolves the MMI to the General purpose radio and preserves day/night use. | — |
| S0321 | 6.4.2.1 | revise | Describe the supplied control's design; “the system must design” assigns a design activity to the system. | provide an on/off control designed to prevent accidental activation or deactivation |
| S0322 | 6.4.2.2 | accept | Preserves facilities for loudspeaker-volume adjustment. | — |
| S0325 | 6.4.2.5 | accept | Preserves protection of both stored numbers and other setup details from accidental changes. | — |
| S0327 | 6.4.3.1 | accept | Preserves the exact ten-minute full-duplex battery threshold and both indications. | — |
| S0328 | 6.4.3.2 | accept | Preserves the user-facing indication of network-service availability. | — |
| S0337 | 7.2.2.4 | revise | Restore the obtained number as the actual call destination. | attempt to establish a call to the obtained number with railway operation priority once an appropriate number has been obtained |
| S0338 | 7.2.2.5 | revise | Resolve the call as the controller call described in 7.2.2.3–6. | provide the user with an audible and visual indication that the call to the controller is proceeding |
| S0339 | 7.2.2.6 | accept | Preserves connection to the controller as the trigger and both user indications. | — |
| S0340 | 7.2.2.7 | accept | Preserves connected-party functional identity, recipient, and availability condition. | — |
| S0341 | 7.2.2.8 | accept | Preserves the alphanumeric identity description and availability condition. | — |
| S0348 | 7.2.2.26 | accept | Preserves entering/leaving shunting mode and its implementation condition. | — |
| S0355 | 7.3.3 | accept | Preserves ruggedness, Operational radio, railway environment, and operational personnel. | — |
| S0357 | 7.3.5 | revise | Describe suitable controls rather than assigning their design to the system. | provide controls designed for use by people wearing gloves |
| S0362 | 7.3.11 | accept | Preserves Operational radio compatibility with a car adapter kit. | — |
| S0364 | 7.4.1.2 | accept | Resolves the Operational radio MMI and preserves day/night suitability. | — |
| S0367 | 7.4.3.2 | revise | Describe the supplied button's design rather than a system design activity. | provide an emergency button designed to avoid accidental use |
| S0368 | 7.4.3.3 | revise | Preserve the conditional protective button design without making the system a designer. | provide a link assurance button designed to avoid accidental use if shunting mode is implemented |
| S0369 | 7.4.3.4 | accept | Preserves the dedicated PTT button, unlike merely a PTT function. | — |
| S0370 | 7.4.4.1 | accept | Preserves the exact ten-minute full-duplex battery threshold and audible/visual alarm. | — |
| S0384 | 9.2.1.1 | accept | Preserves EIRENE users, originating and receiving calls, and functional-number addressing. | — |
| S0385 | 9.2.1.2 | accept | Preserves one unique telephone-number identity for each mobile. | — |
| S0387 | 9.2.2.2 | accept | Preserves every on-train function and unique standard numbering. | — |
| S0389 | 9.2.3.2 | accept | Preserves every on-engine/coach function and unique standard numbering. | — |
| S0394 | 9.2.5.1 | accept | Preserves free configuration within each network's operational responsibility. | — |
| S0399 | 9.4.2 | revise | Identify the local area as a group-call area from the Group numbers section and voice-service definition. | allocate the pre-defined local area for group calls on a bilateral basis in network boundary areas |
| S0400 | 9.5.1 | accept | Preserves authorised EIRENE recipients and callers outside the network. | — |
| S0405 | 10.2.3 | accept | Preserves lowest-priority-first pre-emption ordering. | — |
| S0411 | 10.4.3 | accept | Preserves the link between mobile group deactivation and blocked receipt from that group. | — |
| S0413 | 10.4.5 | accept | Preserves subscribed mobiles, emergency groups, operational condition, and prohibition on deactivation. | — |
| S0418 | 10.6.3 | accept | Preserves railway authority to add restrictions if required. | — |
| S0421 | 11.2.1.1 | revise | Restore the essential distinction between role-based numbers and equipment-bound numbers. | provide an addressing scheme that identifies users by numbers corresponding to their functional roles rather than numbers tied to the terminal equipment they use |
| S0426 | 11.2.1.7 | accept | Preserves setup on one network and cancellation of the same number from another. | — |
| S0444 | 11.3.2.4 | accept | Preserves the registration procedure and means to prevent functional-number misallocation. | — |
| S0451 | 11.3.3.5 | accept | Preserves removal of the displayed number and user notification on the EIRENE mobile. | — |
| S0452 | 11.3.4.1 | accept | Preserves functional-number re-registration after new-network selection and its roaming purpose. | — |
| S0454 | 11.3.4.3 | accept | Preserves new registration details, user display, and completion of automatic re-registration. | — |
| S0455 | 11.4.1 | accept | Preserves function-based routing to a destination number dependent on user location. | — |
| S0459 | 11.4.5 | accept | Preserves availability of location-dependent addressing to all mobiles. | — |
| S0460 | 11.4.6 | accept | Preserves the minimum network-derived location source without excluding additional sources. | — |
| S0471 | 12.4.1 | accept | Preserves the standard interface and both mobile/application-module endpoints. | — |
| S0475 | 13.1.4 | revise | Resolve call type to Railway emergency calls; source context distinguishes Train and Shunting emergency calls. | determine the type of Railway emergency call automatically from the radio's mode of operation |
| S0478 | 13.1.6i | accept | Preserves the appropriate ERTMS/ETCS RBC and Train emergency-call initiation trigger. | — |
| S0479 | 13.1.7 | accept | Preserves all shunting participants in the shunting area as recipients. | — |
| S0480 | 13.1.8 | accept | Preserves automatic priority of the Shunting emergency call over link assurance. | — |
| S0482 | 13.2.2.1 | accept | Preserves simple MMI initiation and the single-action example for Cab/Operational radios. | — |
| S0485 | 13.2.2.3i | revise | Replace “this period” with the thirty-second automatic retry period in 13.2.2.3. | provide the user with an audible and visual indication of connection attempts during the 30-second automatic retry period for an unconnected Railway emergency call |
| S0488 | 13.2.2.5 | revise | Identify the indications as emergency-function activation indications from the warning-stage context. | provide different emergency-function activation indications at the originating and receiving terminals |
| S0489 | 13.2.2.6 | accept | Preserves continuous visual indication, emergency activation, and all receiving terminals. | — |
| S0491 | 13.2.3.1 | accept | Preserves immediate post-warning speech connection and the originator's emergency information. | — |
| S0494 | 13.2.3.3 | revise | Resolve “the information” to the spoken emergency information in 13.2.3.1–2. | deliver spoken emergency information to the same users who received the warning tone |
| S0501 | 13.4.3 | revise | Resolve the confirmation message and call to Railway emergency-call confirmation under 13.4.1. | start the Railway emergency-call confirmation message at the end of the emergency call or when the radio moves out of the call area |
| S0512 | 14.2.2i | revise | Resolve other members as the shunting group; preserve exclusivity and signal-transmission interval. | allow only the link-assurance-signal originator to speak to all other shunting group members using the PTT function during signal transmission |
| S0516 | 14.2.6 | accept | Preserves group-member control of activation/deactivation on Operational radios used for shunting. | — |
| S0517 | 14.2.7 | accept | Preserves link-assurance purpose, the driver recipient, and the member at the head of the movement. | — |

### ERTMS requirements

| Seed ID | Source requirement | Decision | Rationale | Corrected capability if revised |
| --- | --- | --- | --- | --- |
| S0541 | 3.1.1.1a | accept | Preserves provision of ETCS information to the driver for safe train driving. | — |
| S0544 | 3.1.1.10 | accept | Preserves ETCS operation through the stated maximum speed of 500 km/h. | — |
| S0560 | 3.9.1.6 | accept | Preserves current operational status, driver recipient, and DMI. | — |
| S0562 | 3.10.1.3 | accept | Preserves a defined area as the scope of national values. | — |
| S0563 | 3.10.1.6 | accept | Preserves received values' validity even across onboard-equipment switch-off. | — |
| S0565 | 3.10.1.2 | accept | Preserves harmonised defaults, permanent storage, and all onboard equipment. | — |
| S0568 | 4.1.1.4c | accept | Preserves self-test results on the DMI. | — |
| S0569 | 4.1.2.1a | accept | Preserves train-data entry as a movement precondition; “require” expresses that behavior. | — |
| S0571 | 4.1.2.3a | revise | Make clear that the train, rather than only the driver, must be stationary for manual entry/overwrite. | allow the driver to enter or overwrite data manually only when the train is stationary |
| S0572 | 4.1.2.5a | accept | Preserves automatic entry and both railway-management-system and train-memory sources. | — |
| S0577 | 4.1.2.14a | accept | Preserves both driver-identification entry and language selection. | — |
| S0578 | 4.1.2.14b | accept | Preserves the two changeable items and the source's journey qualification on driver identification. | — |
| S0579 | 4.1.2.15 | accept | Preserves successful data entry as the antecedent for shunting or train movements. | — |
| S0582 | 4.1.2.17 | accept | Preserves awakening, failed RBC contact, and requesting contact details from the driver. | — |
| S0584 | 4.1.3.2a | revise | Identify the train as stationary and retain driver-selected transfer to Shunting. | allow transfer to Shunting on the driver's selection only when the train is stationary |
| S0585 | 4.1.3.2c | revise | Resolve “the function” as transfer to Shunting from 4.1.3.2a. | obtain RBC permission for transfer to Shunting when the train operates under RBC control to prevent unauthorised use |
| S0586 | 4.1.3.2d | revise | Identify the permission as the RBC's permission for transfer to Shunting. | indicate received RBC permission for transfer to Shunting to the driver |
| S0590 | 4.1.3.5a | accept | Preserves ETCS supervision of Shunting to the permitted national speed value. | — |
| S0593 | 4.1.3.8a | accept | Preserves the train-stationary-only condition for exiting Shunting. | — |
| S0597 | 4.1.4.3 | accept | Preserves Partial Supervision indication on the DMI. | — |
| S0599 | 4.1.4.4b | revise | Restore Partial Supervision scope from 4.1.4.4a and the section heading. | allow the train to be supervised to a ceiling speed during Partial Supervision |
| S0603 | 4.1.5.1a | accept | Preserves automatic transfer and receipt of movement authority plus all necessary track-to-train information. | — |
| S0605 | 4.1.5.4 | accept | Preserves both speed and distance supervision in Full Supervision. | — |
| S0612 | 4.1.8.1 | accept | Preserves trackside ordering as a sufficient condition for Unfitted operation. | — |
| S0613 | 4.1.8.2 | accept | Preserves driver selection at startup as a condition for Unfitted operation. | — |
| S0619 | 4.1.1.3b | revise | The driver changes the represented adhesion conditions, not physical adhesion; trackside information retains priority. | allow the driver to change adhesion-condition information while giving priority to trackside information |
| S0620 | 4.1.1.4a | accept | Preserves trackside as the sender and speed-profile calculation as the purpose. | — |
| S0622 | 4.1.1.5 | accept | Preserves track-to-train transmission and distinct profiles for specific train categories. | — |
| S0623 | 4.2.2.1 | accept | Preserves the trainborne actor, end-of-authority supervision, and onboard-availability condition. | — |
| S0627 | 4.2.3.2 | revise | Repair “as data National Value” and retain the On Sight section's ceiling-speed scope. | define the ceiling speed level for an On Sight movement authority as a National Value |
| S0628 | 4.2.3.4 | revise | Explicitly identify the train as entering the occupied track, while the system requests acknowledgement. | request driver acknowledgement before the train enters an occupied track |
| S0629 | 4.2.3.6a | revise | Restore On Sight scope so the available-speed-data rule is not generalised to every operating mode. | supervise the train according to available train-speed data during On Sight operation |
| S0637 | 4.3.2.1a | accept | Preserves both emergency and service braking curves based on all relevant data. | — |
| S0638 | 4.3.2.2a | accept | Preserves lower-speed transition, train front, and dynamic profile. | — |
| S0639 | 4.3.2.2b | accept | Preserves higher-speed transition, train rear, and static profile. | — |
| S0641 | 4.3.2.5 | accept | Preserves braking curves' role in meeting train-speed requirements. | — |
| S0644 | 4.3.3.2c | accept | Preserves release-speed indication on the DMI. | — |
| S0647 | 4.3.3.4 | accept | Preserves each railway's ability to choose a different release speed for each signal. | — |
| S0648 | 4.3.4.1 | accept | Preserves ETCS trainborne determination of the entire train's location. | — |
| S0649 | 4.3.4.2 | accept | Preserves the entire train, sending equipment, RBC destination, and RBC-equipped-line condition. | — |
| S0650 | 4.3.4.3 | accept | Preserves odometry error in the train-location calculation. | — |
| S0651 | 4.3.51a | accept | Preserves actual-speed indication on the DMI. | — |
| S0657 | 4.3.7.1 | accept | Preserves supervision against both static and dynamic train-speed profiles. | — |
| S0660 | 4.3.7.4b | revise | Identify the train, rather than the driver, as stationary; do not add “only” because the next source rule permits a national-value exception. | allow the driver to release an ETCS emergency-brake application when the train is stationary |
| S0661 | 4.3.7.4c | accept | Preserves the national-value permission and actual speed strictly below permitted speed. | — |
| S0662 | 4.3.9.1a | accept | Preserves direction comparison and protection against both roll-away and unwanted reverse movements. | — |
| S0663 | 4.3.9.1b | revise | Restore the roll-away/unwanted-reverse context from 4.3.9.1a; the threshold does not refer to arbitrary train travel. | apply the emergency brake with the trainborne equipment after the train travels the distance defined by a national value during roll-away or unwanted reverse movement |
| S0665 | 4.3.9.2 | revise | Identify the emergency brake as the roll-away/reverse intervention in 4.3.9.1a–c. | allow the driver to release the emergency brake applied for roll-away or unwanted reverse movement once the traction unit is at a standstill |
| S0667 | 4.3.9.4 | revise | Resolve “this function” to roll-away and reverse-movement protection; retain every non-leading unit. | disable roll-away and reverse-movement protection in all but the leading traction unit when more than one traction unit is used |
| S0669 | 4.3.10.3 | accept | Preserves sufficient recording accuracy for both ETCS functioning and driving behavior. | — |
| S0678 | 4.4.2.5 | revise | Retain driver ID entry and make the multiple-traction-unit context explicit. | allow the driver to enter the driver ID when using multiple traction units |
| S0680 | 4.5.2.1 | revise | Restore the restricted-authority stop-signal-passing procedure as the scope of the national-value speed constraint. | keep train speed at or below the speed specified by a national value during the procedure for passing a stop signal with restricted movement authority |
| S0681 | 4.5.2.2a | revise | Identify the override as the restricted-authority stop-signal control; preserve the received permission. | allow the driver to select the override control for passing a stop signal with restricted movement authority according to the permission received |
| S0682 | 4.5.2.2b | accept | Preserves protection of the override control against inadvertent operation. | — |
| S0683 | 4.5.2.3 | revise | Preserve the authorised override procedure; unqualified stop-signal passage otherwise conflicts with the separate train-trip rule. | suppress the train-trip function when the train passes a stop signal using the override control under a restricted movement authority |
| S0685 | 4.5.2.5a | revise | Add the source procedure's scope; the source does not specify the special indication's content. | show a special indication on the DMI during the procedure for passing a stop signal with restricted movement authority |
| S0691 | 4.6.4.7 | accept | Preserves receipt of an emergency stop as the trigger for an ETCS emergency-brake command. | — |
| S0695 | 4.6.4.2 | revise | Resolve “the function” as route-suitability protection in 4.6.4.1a–c. | allow the driver to override the route-suitability protection function when the train is stationary |
| S0697 | 4.6.12.1 | accept | Preserves the stop-signal passage trigger and emergency-brake action in the train-trip function. | — |
| S0699 | 4.6.12.3 | revise | Restore the train-trip context so the clause does not require perpetual emergency braking. | apply the emergency brake following a train trip until the traction unit is stationary |
| S0702 | 4.6.12.5b | revise | Resolve acknowledgement as the driver's acknowledgement of the train-trip condition in 4.6.12.4. | allow the train to be driven backwards for a distance defined by a national value after the driver acknowledges the train-trip condition |
| S0707 | 4.8.1.5a | accept | Preserves pantograph and power-supply information displayed by the trainborne equipment on the DMI. | — |
| S0711 | 4.8.8.3 | accept | Preserves driver alerting when a plain-text message appears on the DMI. | — |
| S0713 | 4.8.8.6 | revise | Preserve the character set actually used for plain-text messages, rather than merely providing an unused set. | use a character set that supports different languages for plain-text messages |
| S0717 | 4.8.10.2 | revise | Resolve “information” to brake-inhibition information, including all three types listed in 4.8.10.1. | show information about inhibition of regenerative, eddy current and magnetic shoe brakes on the DMI |
| S0724 | 4.9.10.3 | revise | Identify the target as the RBC's proposed target for cooperative movement-authority revocation. | use the onboard equipment to check acceptance of the new target location proposed by the RBC for cooperative revocation of the movement authority |
| S0727 | 4.9.10.2 | accept | Expands MA correctly and preserves the movement-authority value defining reversing. | — |
| S0734 | 4.9.12.3 | accept | Preserves single-operational-radio handover and possible, not inevitable, performance loss. | — |

## Limitations and integration checks

The source documents contain compressed or awkward wording, duplicated ERTMS requirement numbers, and underspecified details. Seed ID plus source corpus and section identifies a row reliably; a bare ERTMS requirement number does not. Source terms such as “special indication”, “other interfaces”, “as far as possible”, and “certain distance defined by national value” remain unspecified where the source is unspecified. No indication content, interface type, feasibility guarantee, or numerical national value was invented. The power-on self-test comparison uses the surrounding startup and radio-test requirements to resolve the source sentence's attachment ambiguity. The source's attachment of “during a journey” to driver-identification changes is retained without extending that condition to Train Running Number changes.

No item remains without a proposed disposition. This does not remove source underspecification or establish expert/human agreement. Several seeds concern hardware properties, design assurance, or operator interaction; their generic “system” phrasing represents those source capabilities and does not turn them into evaluated runtime functionality.

Before rebuilding the benchmark, apply the complete replacement cells for `revise` rows and retain the reviewed proposal for `accept` rows. Check exact coverage of the selected seed IDs and render all modality prefixes with the resulting clauses. Preserve the full shutdown-housekeeping capability even if it exceeds a prior automatic length heuristic. This review did not modify the source CSV, proposal CSV, generated outputs, code, profiles, or manuscript, and made no provider calls.

Structural verification passed: 180 unique review IDs exactly match both the proposal and candidate tables; all source requirement identifiers match; all 51 revision cells contain replacements and all 129 accept cells are marked with a dash. The longest replacement contains 29 whitespace-delimited words. The review's Markdown has no trailing whitespace.
