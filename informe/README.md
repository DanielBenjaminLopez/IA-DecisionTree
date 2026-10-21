# Documentación del Informe Académico (Formato Springer LNCS)

La consigna de la cátedra establece que el informe debe seguir estrictamente el formato **LNCS (Lecture Notes in Computer Science)** de **Springer Verlag** y no exceder las **12 carillas**.

---

## 🛠️ Opción 1: Redacción en LaTeX (Recomendada)

LaTeX es el estándar recomendado para artículos científicos y el formato natural de Springer LNCS.

### Archivos incluidos:
- **`informe_fuente.tex`:** Estructura completa del informe con todas las secciones exigidas por la cátedra (Introducción, Algoritmos, Solución implementada, Simulaciones, Conclusiones, Referencias).
- **`referencias.bib`:** Archivo BibTeX con las citas bibliográficas obligatorias (Quinlan 1993, Liu 2011, Breiman 2001, Scikit-learn 2011).
- **`figures/`:** Directorio para guardar gráficos, matrices de confusión y diagramas generados durante los experimentos.
- **`informe_LNCS.pdf`:** Versión final compilada que se entregará a la cátedra.

### Cómo editar y compilar:
1. **En Overleaf (Recomendado para trabajo colaborativo grupal):**
   - Crea un nuevo proyecto en [Overleaf](https://www.overleaf.com/).
   - Sube los archivos `informe_fuente.tex`, `referencias.bib` y la carpeta `figures/`.
   - Para compilar correctamente LNCS en Overleaf, puedes subir el archivo de clase oficial `llncs.cls` y el estilo `splncs04.bst` disponibles en la web de Springer.
2. **Localmente (con texlive o miktex):**
   ```bash
   pdflatex informe_fuente.tex
   bibtex informe_fuente
   pdflatex informe_fuente.tex
   pdflatex informe_fuente.tex
   ```

---

## 📝 Opción 2: Redacción en Microsoft Word (Alternativa permitida)

Si el equipo prefiere trabajar en Word, Springer provee una plantilla oficial `.docx` que replica el formato LNCS:

1. **Descarga de la plantilla Word de Springer LNCS:**
   - [Guía y plantillas de actas de congresos de Springer](https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines)
   - Buscar la sección *"Proceedings Guidelines for Authors"* > *"Word Template"*.
2. **Requisitos de formato a mantener en Word:**
   - Tipografía: Times New Roman o similar, cuerpo 10pt (interlineado simple).
   - Márgenes: 5.2 cm superior, 5.7 cm inferior, 4.4 cm laterales (formato estándar LNCS).
   - No exceder las **12 carillas**.
   - Citar todas las fuentes en formato numérico o autor-año según la plantilla.
   - Exportar a PDF como `informe_LNCS.pdf` y guardar la fuente Word en esta carpeta (ej. `informe_fuente.docx`).

---

## ⚠️ Aspectos evaluados por la cátedra
- **Prolijidad, redacción y ortografía:** Crucial para la nota final.
- **Integridad y citas:** Toda fuente utilizada debe estar debidamente citada. Evitar plagio estricto.
