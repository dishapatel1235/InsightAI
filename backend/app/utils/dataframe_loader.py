import pandas as pd


class DataFrameLoader:
    @staticmethod
    def load_dataframe(file_path: str) -> pd.DataFrame:
        lower_path = file_path.lower()

        if lower_path.endswith(".csv"):
            # Try UTF-8 with BOM first, then UTF-8, then Latin-1 for Excel-exported CSVs
            encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
            for enc in encodings:
                try:
                    return pd.read_csv(file_path, encoding=enc, low_memory=False)
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception as e:
                    # If parsing error, try with engine='python'
                    try:
                        return pd.read_csv(file_path, encoding=enc, engine="python")
                    except Exception:
                        raise e
            # Final fallback
            return pd.read_csv(file_path, encoding_errors="replace", low_memory=False)

        elif lower_path.endswith((".xlsx", ".xls")):
            return pd.read_excel(file_path)

        else:
            # Fallback: attempt to load as CSV anyway
            try:
                return pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace", low_memory=False)
            except Exception:
                raise ValueError("Unsupported file format. Please upload a valid CSV (.csv) or Excel (.xlsx) file.")
