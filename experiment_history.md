# Dziennik Eksperymentów i Obserwacji (Experiment History)

Niniejszy dokument stanowi chronologiczny zapis prac nad architekturą modelu, napotkanych problemów technicznych oraz wniosków wyciągniętych z cząstkowych wyników ewaluacji.

---

## FAZA 1: Problemy z TRL i DPO (Direct Preference Optimization)
**Założenie:** Zastosowanie biblioteki `trl` (DPOTrainer) do optymalizacji preferencji w modelu PLT5 (Seq2Seq).

**Obserwacje i Problemy:**
1. **Błędy Architektur:** Biblioteka `trl` w nowszych wersjach (np. 1.5.1) milcząco zakładała, że model jest architektury Causal LM (Decoder-only, jak GPT). Prowadziło to do błędnego tokenizowania wejść (konkatenacja `prompt` + `target`) i usuwania kolumny `labels`, co powodowało twarde błędy `ValueError` z poziomu PyTorcha, ponieważ model T5 fizycznie wymagał etykiet (labels) do przeprowadzenia propagacji wstecznej.
2. **Piekło Zależności (API Drift):** Próba downgrade'u biblioteki do wersji rzekomo obsługującej Seq2Seq (`0.8.6`) wywołała lawinę błędów kompatybilności z najnowszym `transformers` (zmiana `tokenizer` na `processing_class`, zmiana liczby argumentów w funkcjach pętli trenującej). Nawet po zastosowaniu rozległych monkey-patchy ewaluacja modelu sypała błędami generatora.

**Decyzja:** Całkowite porzucenie biblioteki `trl` i implementacja własna.

---

## FAZA 2: Implementacja Autorskiego ORPO (Odds Ratio Preference Optimization)
**Założenie:** Napisanie niestandardowego `CustomORPOTrainer` dziedziczącego po wbudowanym, stabilnym `Seq2SeqTrainer`.

**Zalety rozwiązania:**
1. **Brak Modelu Referencyjnego:** W przeciwieństwie do DPO, ORPO wplata optymalizację preferencji bezpośrednio w Supervised Fine-Tuning (SFT). Zaoszczędziło to 50% VRAMu na GPU (RTX 3050).
2. **Natywna Zgodność:** Własny `ORPODataCollator` precyzyjnie kontrolował wymiary tensorów `input_ids`, `attention_mask`, `chosen_labels` i `rejected_labels`, pozwalając T5 generować wyniki i ewaluować się bez zakłóceń.

---

## FAZA 3: Pierwsze Prawdziwe Treningi i "Czerwone Flagi"
Po udanym wdrożeniu ORPO przeprowadzono pierwsze pełne treningi na małym modelu (`plt5-small`).

### Wyniki Wariantu A (Tylko Błędy - Samo SFT)
* **False Positive Rate (FPR):** 9.23% (Model psuł 9% poprawnych zdań - silny over-correction bias).
* **True Positive Rate (TPR):** 2.31% (Model bardzo słabo radził sobie z korektą złożonych błędów interferencyjnych).
* **BERTScore:** ~0.95 (Model rzadko halucynował, zachowywał sens, ale nie umiał poprawnie użyć gramatyki).

### Wyniki Wariantu B (25% Zdań Poprawnych - ORPO) -> PARADOKS ORPO
Oczekiwano zdań poprawnych jako kotwicy dla niskiego FPR. Zamiast tego zaobserwowano katastrofę:
* **FPR (Pepsute zdania):** 93.85% (Kategoryczny kolaps).
* **TPR (Poprawione błędy):** 0.77%.
* **BERTScore:** Spadek do 0.88 (Model zaczął aktywnie halucynować i zmieniać sens słów).

**Odkrycie - "Paradoks Tożsamości ORPO":** 
W algorytmie Preference Optimization model jest matematycznie karany za wygenerowanie tokenów ze zbioru `rejected`. W przypadku zdań w 100% poprawnych (Identity Translations) ustawiliśmy `chosen == rejected`. Zatem model był "karany" za wygenerowanie dokładnie tego samego, poprawnego zdania. Aby zminimalizować błąd (loss), zaczął modyfikować poprawne zdania na siłę.

---

## FAZA 4: Maskowanie i Zwiększenie Pojemności (Obecna Architektura)
Na podstawie powyższych obserwacji, przebudowano architekturę i konfigurację:

### 1. Matematyczne Maskowanie Tożsamości (The Identity Mask)
W funkcji straty wprowadzono mnożnik z flagi `is_error`. Jeśli zdanie jest tożsamościowe (`is_error=0`), kara ORPO jest mnożona przez 0. Model na tych zdaniach uczy się **tylko** za pomocą SFT (Cross-Entropy), co naprawiło "Paradoks ORPO" i pozwoliło mu bezpiecznie kopiować poprawny tekst (FPR spadł do **1.54%**).

### 2. Zwiększenie Pojemności Lingwistycznej Modelu
Zauważono, że TPR rzędu 2% to efekt zbyt małego modelu. Zadania typu Transfer Learning na ukraińskiej/rosyjskiej składni przekraczają możliwości 60-milionowego T5.
* Zmiana modelu z `plt5-small` na **`plt5-base`** (~220M parametrów).
* Zwiększenie modułów Low-Rank Adaptation (`lora_r=64`), aby model miał więcej plastycznych parametrów.

---

## FAZA 5: Skalowanie do PLT5-Base i Reranking z modelem Herbert
Przeniesienie eksperymentów na model bazowy oraz zastosowanie dodatkowych modułów oceniających przyniosło finalne, kompletne wnioski.

### 1. Skalowanie danych (10k vs. 50k SFT)
* Trening na 50k danych (Pipeline 3) wykazał znaczącą redukcję straty walidacyjnej (Train Loss spadł z **1.0668** do **0.5686**) i skok EM o **+10.36%** na zbiorze syntetycznym.
* Na autentycznym zbiorze (OOD) model bazowy 50k osiągnął **27.69% EM** oraz F0.5 = **0.2385**. Nadal jednak dominował copying bias (FNR ~89%).

### 2. Rerankowanie Płynności (Herbert Reranker - Pipeline 4)
* Zastosowanie modelu Herbert do wyboru najlepszej hipotezy z Beam Search przełamało copying bias, **podwajając TPR z 10.77% do 23.46%**.
* Spowodowało to jednak skok **FPR do 24.62%** (nadkorekcja). Wynika to z faktu, że model Herbert optymalizuje wyłącznie gramatyczność i naturalność polskiego sformułowania (PLL), ignorując wierność semantyczną oryginału.

### 3. Wpływ Rozmiaru Wiązki (Beam Size 3 vs 5)
* Wiązka o rozmiarze 3 (Pipeline 3 + Rerank) redukuje FPR (**16.92%**), ale nie generuje trudniejszych odmian morfologicznych (np. rodzaj gramatyczny w kategorii *Gender* miał TPR = **0.00%**).
* Wiązka o rozmiarze 5 (Pipeline 4) dostarcza do rerankera poprawne morfologicznie formy (Gender TPR wzrósł do **28.57%**, a Case TPR do **42.86%**), lecz zwiększenie przestrzeni generuje również więcej zniekształceń, podnosząc FPR do **24.62%**.
* **Wniosek:** Skuteczne wdrożenie rerankera wymaga filtrowania progiem zysku płynności ($\Delta PLL$) w celu odrzucenia fałszywych poprawek na zdaniach poprawnych.