Projekt NLP: Korekta Interferencji Językowej L1 (Grammatical Error Correction)

**Pytanie badawcze:** *Czy polski model językowy typu Encoder-Decoder (PLT5) jest w stanie zidentyfikować błędy wynikające z interferencji języka ojczystego (kalki przyimkowe, błędna rekcja, fałszywi przyjaciele) i zrekonstruować zdanie do naturalnej formy?*

**Cel eksperymentu:** Celem projektu jest zbadanie i rozwiązanie problemów "nadmiernej korekty" (over-correction) oraz "ślepego kopiowania" (copying bias) w modelach Seq2Seq dla języka polskiego. Ewaluujemy model w dwóch skalach rozmiaru i zbioru danych:
1. **Model Mały (`plt5-small`, 60M parametrów):** Analizujemy wpływ algorytmów optymalizacji preferencji (ORPO) i proporcji zdań poprawnych (Identity Translations) na zbiorze 10k par.
2. **Model Bazowy (`plt5-base`, 275M parametrów):** Skalujemy model do wersji base na zbiorach 10k i 50k par.
3. **Wyszukiwanie i Reranking (Fluency Reranker):** Wprowadzamy model *Herbert* jako zewnętrzny oceniacz płynności języka polskiego na wyjściach Beam Search w celu przełamania copying bias.

Porównujemy następujące potoki (Pipelines) i warianty:
*   **Pipeline 1 (Ablacja Małego Modelu):**
    *   **Wariant A (Tylko Błędy):** Trening SFT wyłącznie na błędach.
    *   **Wariant B (SFT / ORPO + Identity):** Trening na miksie błędów i zdań poprawnych (10% i 30% Identity Translations).
    *   **Wariant C (Transfer Learning):** Sprawdzenie uogólniania modelu na niewidzianych klasach błędów.
*   **Pipeline 2 (Base SFT 10k):** Model `plt5-base` trenowany na 10k danych z 10% udziałem zdań poprawnych.
*   **Pipeline 3 (Base SFT 50k):** Model `plt5-base` trenowany na 50k danych z 10% udziałem zdań poprawnych.
*   **Pipeline 4 (Base SFT 50k + Herbert Rerank):** Rerankowanie wyników generowania z Pipeline 3 za pomocą polskiego modelu Herbert w celu maksymalizacji wskaźnika poprawek.
*   **Baseline (Do-Nothing):** Kopiowanie wejścia bezpośrednio na wyjście (brak jakichkolwiek modyfikacji).

---

## 1. Tworzenie zbioru danych i podział na eksperymenty
### Źródła Danych
1.  **Korpus Bazowy:** Zbiór zdań w języku polskim został pozyskany z dumpu polskiej **Wikipedii** (`wikimedia/wikipedia`, `20231101.pl`) dostępnego na Hugging Face. Zdania te stanowią podstawę, do której wstrzykiwane są błędy.
2.  **Słownik Morfologiczny:** Do poprawnego odmiany słów (zarówno przy tworzeniu błędów, jak i ich naprawie) wykorzystano słownik **PoliMorf**, opracowany przez Instytut Podstaw Informatyki PAN. Jest to kluczowe narzędzie do zachowania spójności gramatycznej.
3.  **Baza Fałszywych Przyjaciół:** Błędy leksykalne (fałszywi przyjaciele) zostały zebrane z kilku źródeł: 
    *   **Wiktionary:** Automatycznie scrapowano tabele fałszywych przyjaciół między językiem polskim a rosyjskim, ukraińskim i angielskim.
    *   **Zbiór Kuratorski:** Ręcznie przygotowana lista najczęstszych pomyłek leksykalnych, nieuwzględnionych w Wiktionary.
4.  **Reguły Błędów:** Błędy składniowe (rekcja przyimkowa, błędny przypadek) oraz błędy rodzaju gramatycznego zostały zdefiniowane w postaci statycznych reguł w kodzie, bazując na typowych błędach obserwowanych u osób rosyjsko- i ukraińskojęzycznych.

### Proces Syntezy
Skrypt `prepare_data.py` orkiestruje cały proces, który obejmuje pobranie, przetworzenie i syntezę danych. Zdania z Wikipedii są przetwarzane przez model `spaCy` w celu analizy morfologicznej. Następnie, z określonym prawdopodobieństwem, w zdaniach wstrzykiwane są błędy jednego z typów (leksykalne, składniowe, rodzajowe). Każda para składa się z oryginalnego, poprawnego zdania (`label=0`) oraz jego zepsutej wersji (`label=1`). Dodatkowo, część poprawnych zdań pozostaje nietknięta, tworząc doomed **Identity Translations** (`is_error=0`), które są kluczowe dla Wariantu B.

---

## 1. Tworzenie zbioru danych i podział na eksperymenty
### Źródła Danych
1.  **Korpus Bazowy:** Zbiór zdań w języku polskim został pozyskany z dumpu polskiej **Wikipedii** (`wikimedia/wikipedia`, `20231101.pl`) dostępnego na Hugging Face. Zdania te stanowią podstawę, do której wstrzykiwane są błędy.
2.  **Słownik Morfologiczny:** Do poprawnego odmiany słów (zarówno przy tworzeniu błędów, jak i ich naprawie) wykorzystano słownik **PoliMorf**, opracowany przez Instytut Podstaw Informatyki PAN. Jest to kluczowe narzędzie do zachowania spójności gramatycznej.
3.  **Baza Fałszywych Przyjaciół:** Błędy leksykalne (fałszywi przyjaciele) zostały zebrane z kilku źródeł: 
    *   **Wiktionary:** Automatycznie scrapowano tabele fałszywych przyjaciół między językiem polskim a rosyjskim, ukraińskim i angielskim.
    *   **Zbiór Kuratorski:** Ręcznie przygotowana lista najczęstszych pomyłek leksykalnych, nieuwzględnionych w Wiktionary.
4.  **Reguły Błędów:** Błędy składniowe (rekcja przyimkowa, błędny przypadek) oraz błędy rodzaju gramatycznego zostały zdefiniowane w postaci statycznych reguł w kodzie, bazując na typowych błędach obserwowanych u osób rosyjsko- i ukraińskojęzycznych.

### Proces Syntezy
Skrypt `prepare_data.py` orkiestruje cały proces, który obejmuje pobranie, przetworzenie i syntezę danych. Zdania z Wikipedii są przetwarzane przez model `spaCy` w celu analizy morfologicznej. Następnie, z określonym prawdopodobieństwem, w zdaniach wstrzykiwane zijn błędy jednego z typów (leksykalne, składniowe, rodzajowe). Każda para składa się z oryginalnego, poprawnego zdania (`label=0`) oraz jego zepsutej wersji (`label=1`). Dodatkowo, część poprawnych zdań pozostaje nietknięta, tworząc doomed **Identity Translations** (`is_error=0`), które są kluczowe dla Wariantu B.

---

## Ewaluacja na Zbiorze Zewnętrznym (Out-of-Distribution)
Oprócz ewaluacji na syntetycznym zbiorze testowym, kluczowym elementem jest sprawdzenie, jak model radzi sobie ze zdaniami, których nigdy nie widział – takimi, które naśladują autentyczne błędy ludzkie. W tym celu wykorzystano model Gemini 3.5 Flash do wygenerowania zbioru `external_validation_dataset.json`.

Wygenerowany zbiór został **poddany ręcznej weryfikacji**. Odfiltrowano i poprawiono halucynacje LLM-a, pozostawiając jedynie te zdania, które nieco odpowiadają rzeczywistej interferencji języków L1 (rosyjski/ukraiński) u osób uczących się języka polskiego.

---

### Wstępne wnioski - Wariant A
**Zjawisko Over-Correction:** model trenowany wyłącznie na parach z błędami wyrobił w sobie silny *bias* zakładający, że każde wejściowe zdanie jest wadliwe. W efekcie, gdy podajemy mu poprawne zdanie (Identity Test: np. "Chłopiec uśmiechnął się szeroko na widok ciasta"), model bardzo często na siłę próbuje coś podmienić, psując zupełnie poprawne zdanie.

---

### Wstępne wnioski - Wariant B (Porównanie SFT vs. ORPO & Skalowanie)

1.  **Analiza SFT vs. ORPO (plt5-small):**
    *   **Stabilność SFT:** Standardowy trening SFT z regularną tożsamością (10% oraz 30% Identity) wykazał się bardzo dobrą kontrolą błędów nadkorekcji. Zarówno dla 10%, jak i 30% Identity, wskaźnik FPR (odsetek zepsutych zdań poprawnych) utrzymał się na poziomie **1.54%** (dla porównania, Wariant A bez Identity psuł **3.08%** poprawnych zdań).
    *   **Instabilność ORPO:** Zastosowanie algorytmu Odds Ratio Preference Optimization na modelu małym (60M parametrów) dało negatywne rezultaty. Wskaźnik FPR wzrósł do **6.15%** (dla 10% Identity) oraz **4.62%** (dla 30% Identity), podczas gdy czułość (TPR) utrzymała się na bazowym poziomie **3.46%**.
    *   **Wniosek:** Małe modele nie posiadają wystarczającej pojemności (capacity) do stabilnego uczenia się z kary Odds Ratio. ORPO próbuje jednocześnie oddalać reprezentacje błędne i przyciągać poprawne, co przy małej liczbie parametrów prowadzi do destabilizacji gradientów i chaotycznego generowania modyfikacji.

2.  **Efekt Skalowania Modelu (plt5-small -> plt5-base):**
    *   Przejście z modelu małego na model bazowy (`plt5-base`, 275M parametrów) przy zachowaniu 10k danych treningowych (Pipeline 2) dało ogromny skok jakościowy. Czułość TPR wzrosła z **3.85%** do **7.69%**, a wskaźnik F0.5 skoczył z **0.1144** do **0.2170** (prawie dwukrotny wzrost).
    *   Większy model znacznie lepiej radzi sobie ze złożoną strukturą rekcji przyimkowej i zależnościami składniowymi języka polskiego.

3.  **Efekt Skalowania Zbioru Treningowego (10k -> 50k):**
    *   Zwiększenie zbioru danych do 50k par (Pipeline 3) przy użyciu modelu bazowego zminimalizowało stratę treningową SFT do poziomu **0.5686** (w porównaniu do **1.0668** przy 10k danych).
    *   Synthetic EM osiągnął poziom **51.96%** (+10.36% poprawy w stosunku do Pipeline 2). Na autentycznym zbiorze ludzkim (OOD) czułość TPR wzrosła do **10.77%**, a wskaźnik F0.5 osiągnął **0.2385**.
    *   **Wniosek:** Skalowanie danych jest niezbędne do pokrycia rzadkich form wyrazowych i zróżnicowanych kontekstów fałszywych przyjaciół (false friends).

    ---

    ### Wstępne wnioski - Wariant C
**Transfer Learning:** Sprawdzenie zdolności modelu do uogólniania koncepcji korekty na nieznane klasy błędów.

---

# CZĘŚĆ III: Podsumowanie Wyników i Analiza Końcowa

## 1. Tabela Wyników Globalnych i Szczegółowa Ewaluacja
Wszystkie ewaluowane potoki (Pipelines 1–4) wraz z bazą odniesienia (Baseline Do-Nothing) i wariantami rerankowanymi za pomocą modelu *Herbert* zostały zestawione w powyższych tabelach. Poniżej przedstawiono szczegółową analizę naukową, techniczną oraz architektoniczną uzyskanych rezultatów.

---

## 2. Wnioski i Analiza Naukowa Eksperymentów

### A. Mały Model (`plt5-small`, 60M) i Paradoks Tożsamości ORPO
1. **Copying Bias w SFT (Wariant A):**
   Trening SFT wyłącznie na zdaniach z błędami utrwala u modelu skłonność do nadkorekcji. Model osiąga wysoki wskaźnik fałszywych alarmów (FPR = **9.23%**), psując poprawne wejścia, oraz wykazuje skrajnie niską czułość (TPR = **3.46%**), bezrefleksyjnie kopiując błędną strukturę wejściową.
2. **Paradoks Tożsamości (Identity Paradox) w ORPO (Wariant B):**
   Zastosowanie algorytmu optymalizacji preferencji (ORPO) na zdaniach poprawnych (Identity Translations), gdzie zdefiniowano `chosen == rejected` (brak błędów do odrzucenia), matematycznie karze model za wygenerowanie poprawnych tokenów. Dążąc do minimalizacji straty, model zaczął celowo zniekształcać poprawne zdania, co zaowocowało katastrofalnym skokiem **FPR do 93.85%**.
3. **Rozwiązanie (Maska Tożsamości):**
   Wprowadzenie mnożnika `is_error` do funkcji straty ORPO (Identity Mask) wyłączyło karę preferencji dla zdań bezbłędnych. Ustabilizowało to FPR wariantu ORPO na niskim poziomie **1.54%**, jednak ze względu na mały rozmiar modelu (60M parametrów), wskaźnik TPR pozostał na niskim poziomie.

### B. Skalowanie Danych i Pojemności Modelu (`plt5-base`, 10k vs 50k)
Skalowanie modelu do wersji bazowej (~220M parametrów) oraz rozbudowa danych treningowych do 50k przyniosły kluczowe korzyści:
* **Spadek Loss:** Strata treningowa została zredukowana z **1.0668** (base 10k) do **0.5686** (base 50k).
* **EM na zbiorze testowym:** Syntetyczna dokładność Exact Match wzrosła o **+10.36%** (z 41.60% do 51.96%).
* **Ewaluacja OOD (Human Eval):** Dokładność uogólniania na autentycznym zbiorze wzrosła do **27.69% EM**, a metryka F0.5 poprawiła się do **0.2385**.
* Model SFT 50k nadal jednak wykazuje silny *copying bias* (wskaźnik FNR wynosi **~89%**), co oznacza, że samo skalowanie parametrów bez zewnętrznej weryfikacji nie wystarcza do eliminacji pasywnego kopiowania.

### C. Fluency Reranker na Wszystkich Eksperymentach (Model Herbert)
* **Generalizacja Rerankingu:** Zewnętrzny rerankujący model płynności *Herbert* został nałożony post-hoc na **wszystkie** wcześniejsze warianty z Pipeline 1, 2 oraz 3, aby ocenić jego zachowanie niezależnie od rozmiaru modelu bazowego i trybu treningu.
* **Złamanie Copying Bias:** We wszystkich wariantach dodanie rerankera przełamało tendencję do ślepego kopiowania, **podwajając czułość (np. TPR z 10.77% do 23.46% dla Pipeline 3)**.
* **Dryf Semantyczny i FPR Surge:** Z powodu braku oceny zgodności semantycznej przez reranker, wskaźnik fałszywych alarmów (FPR) wzrósł we wszystkich wariantach (np. z **4.62% do 24.62%**). Model Herbert wybierał zdania o najwyższym prawdopodobieństwie językowym (PLL), często zastępując poprawne, rzadziej spotykane struktury wejściowe innymi, bardziej typowymi, lecz niezgodnymi semantycznie.

### D. Kompromis Beam Search (Wiązka 3 vs. 5)
* **Wiązka = 3:** Daje niższy FPR (**16.92%**), lecz jest zbyt wąska, aby model wygenerował poprawne formy dla skomplikowanych kategorii (np. TPR dla kategorii *Gender* wyniósł **0.00%**).
* **Wiązka = 5:** Umożliwia wygenerowanie poprawnych poprawek gramatycznych, przywracając TPR dla kategorii *Gender* (**28.57%**) oraz *Case* (**42.86%**) oraz *Typos* (**41.67%**), kosztem podbicia globalnego FPR do **24.62%** ze względu na obecność płynnych, lecz błędnych alternatyw w szerszej przestrzeni.
* **Rekomendacja:** Zastosowanie rerankera w środowisku produkcyjnym wymaga wdrożenia progu decyzyjnego ($\Delta PLL$) w celu odrzucania poprawek o niskim zysku płynności.

---

## 3. Ograniczenia Metodologiczne i Techniczne (Apendyks Diagnostyczny)

W trakcie realizacji projektu zidentyfikowano kluczowe wyzwania techniczne i architektoniczne, które wpłynęły na wyniki eksperymentów i powinny zostać uwzględnione w przyszłych iteracjach badawczych:

1. **Brak Normalizacji Długości Sekwencji w ORPO:**
   W implementacji obliczania prawdopodobieństw logarytmicznych partii danych (`get_batch_logps` w module Custom ORPO) zastosowano sumowanie log-probów tokenów zamiast średniej. Powoduje to, że dłuższe zdania generują nieproporcjonalnie większe wartości ujemne log-probów, co zaburza proporcję szans (Odds Ratio) i wymusza bardzo niską wartość współczynnika `orpo_beta = 0.05` dla zachowania stabilności uczenia.
2. **Ograniczenia Regułowej Syntezy Danych (PoliMorf Gender Mismatch):**
   Algorytm wstrzykiwania "fałszywych przyjaciół" (`_inject_false_friend` w module syntezy) wyszukuje formy deklinacyjne w bazie PoliMorf na podstawie rodzaju gramatycznego oryginalnego polskiego słowa. W przypadkach, gdy słowo zapożyczone ma inny rodzaj (np. polskie *"kanapa"* - rodzaj żeński vs. rosyjskie *"диwan"* - rodzaj męski), słownik morfologiczny zwraca pusty wynik. Powoduje to ciche pomijanie wielu potencjalnych kalk leksykalnych w zbiorze treningowym.
3. **Brak Kaskadowego Uzgodnienia Przypadku w Mutacji Rzeczowników:**
   Wstrzykiwanie błędów przypadków rzeczowników (`_inject_case`) modyfikuje przypadek wyłącznie samego rzeczownika, nie kaskadując tej zmiany na powiązane przymiotniki i określniki. Tworzy to nienaturalne, sztuczne konstrukcje syntaktyczne (np. przymiotnik w narzędniku + rzeczownik w bierniku: *"interesuję się polską literaturę"*), które nie odzwierciedlają wiernie rzeczywistych błędów popełnianych przez uczniów L2.

---


---
## Podziękowania i licencje

W tym projekcie wykorzystano następujące zasoby, których autorom serdecznie dziękujemy:

*   **PoliMorf:** Słownik morfologiczny języka polskiego, opracowany przez Instytut Podstaw Informatyki Polskiej Akademii Nauk. Dostępny na licencji [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
    *   Woliński, M. (2021). *PoliMorf: a (not so) new morphological dictionary of Polish*. Pozyskano z [http://dsmodels.nlp.ipipan.waw.pl/](http://dsmodels.nlp.ipipan.waw.pl/)

*   **Wikipedia (Polska):** Korpus tekstowy użyty jako podstawa do syntezy danych treningowych, pobrany za pośrednictwem Hugging Face (`wikimedia/wikipedia`). Treści dostępne na licencji CC BY-SA 3.0.

*   **Wiktionary:** Dane dotyczące fałszywych przyjaciół zostały pozyskane z angielskiego Wiktionary, które jest dostępne na licencji [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

Wszystkie modele i zbiory danych zostały użyte zgodnie z ich odpowiednimi licencjami.
