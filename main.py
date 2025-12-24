import os
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

DATA_DIR = Path("data")

st.set_page_config(page_title="고등학교 통계 대시보드", layout="wide")
st.title("고등학교 별 통계 그래프 대시보드")

# -----------------------------
# Utilities
# -----------------------------
def read_csv_kr(path: Path) -> pd.DataFrame:
    """Try common Korean encodings."""
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    # 마지막 시도: 기본
    return pd.read_csv(path)

def guess_datetime_column(df: pd.DataFrame):
    """Heuristic to find a datetime-like column."""
    candidates = []
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["date", "time", "datetime", "일자", "날짜", "시간", "측정일", "측정시간"]):
            candidates.append(c)

    # 후보가 있으면 우선 사용, 없으면 object 컬럼 중 datetime 변환 가능한 컬럼 탐색
    cols_to_try = candidates if candidates else list(df.columns)

    best_col = None
    best_ratio = 0.0
    for c in cols_to_try:
        if df[c].dtype == "object" or "datetime" in str(df[c].dtype).lower():
            parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            ratio = parsed.notna().mean()
            if ratio > best_ratio and ratio >= 0.6:  # 60% 이상 파싱 가능하면 채택
                best_col = c
                best_ratio = ratio
    return best_col

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert possible numeric columns stored as strings (e.g., '1,234')."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            s = out[c].astype(str).str.replace(",", "", regex=False).str.strip()
            # 숫자 변환 성공 비율이 높으면 숫자로 변환
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().mean() >= 0.8 and num.notna().sum() >= 5:
                out[c] = num
    return out

@st.cache_data(show_spinner=False)
def load_environment_data():
    """Load and merge environment CSV files with a school column."""
    if not DATA_DIR.exists():
        return pd.DataFrame(), {}

    csv_files = sorted(DATA_DIR.glob("*환경데이터*.csv"))
    school_to_path = {}
    frames = []

    for f in csv_files:
        # 학교명 추정: 파일명에서 '_환경데이터' 앞부분
        m = re.match(r"(.+?)_?환경데이터", f.stem)
        school = m.group(1) if m else f.stem
        school_to_path[school] = f

        df = read_csv_kr(f)
        df = coerce_numeric(df)
        df["학교"] = school
        df["__source_file__"] = f.name
        frames.append(df)

    if not frames:
        return pd.DataFrame(), {}

    merged = pd.concat(frames, ignore_index=True, sort=False)
    return merged, school_to_path

@st.cache_data(show_spinner=False)
def load_growth_xlsx():
    """Load xlsx if exists."""
    if not DATA_DIR.exists():
        return pd.DataFrame(), None

    xlsx_files = sorted(DATA_DIR.glob("*.xlsx"))
    if not xlsx_files:
        return pd.DataFrame(), None

    # 첫 번째 xlsx를 기본으로 사용 (원하면 여러 개 선택 UI로 확장 가능)
    xlsx_path = xlsx_files[0]
    try:
        df = pd.read_excel(xlsx_path)  # openpyxl 필요
        df = coerce_numeric(df)
        return df, xlsx_path.name
    except Exception:
        return pd.DataFrame(), xlsx_path.name


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("설정")

dataset = st.sidebar.radio(
    "데이터 선택",
    ["환경데이터 (CSV)", "생육결과 (XLSX)"],
    index=0,
)

# -----------------------------
# Main: Environment CSV
# -----------------------------
if dataset == "환경데이터 (CSV)":
    env_df, school_map = load_environment_data()

    if env_df.empty:
        st.warning("data/ 폴더에서 '*환경데이터*.csv' 파일을 찾지 못했습니다. 파일명을 확인해 주세요.")
        st.stop()

    # 시간 컬럼 추정
    dt_col = guess_datetime_column(env_df)

    schools = sorted(env_df["학교"].dropna().unique().tolist())
    view_mode = st.sidebar.radio("보기 방식", ["학교별 보기", "학교 비교"], index=0)

    # 수치형 컬럼 후보
    numeric_cols = env_df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["__source_file__"]]  # 안전 필터

    if not numeric_cols:
        st.error("수치형 컬럼(그래프로 그릴 값)을 찾지 못했습니다. CSV 컬럼 구성을 확인해 주세요.")
        st.dataframe(env_df.head(50))
        st.stop()

    metric = st.sidebar.selectbox("지표(컬럼) 선택", numeric_cols)

    # 날짜 필터 (가능한 경우)
    if dt_col:
        tmp = env_df.copy()
        tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors="coerce", infer_datetime_format=True)
        tmp = tmp.dropna(subset=[dt_col])

        if tmp.empty:
            dt_col = None
        else:
            min_d = tmp[dt_col].min().date()
            max_d = tmp[dt_col].max().date()
            date_range = st.sidebar.date_input("기간", value=(min_d, max_d), min_value=min_d, max_value=max_d)
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_d, end_d = date_range
            else:
                start_d, end_d = min_d, max_d
            env_df = tmp[(tmp[dt_col].dt.date >= start_d) & (tmp[dt_col].dt.date <= end_d)].copy()

    st.subheader("데이터 미리보기")
    st.caption(f"총 행 수: {len(env_df):,} | 파일 수: {env_df['__source_file__'].nunique():,} | 시간컬럼: {dt_col if dt_col else '미탐지'}")
    st.dataframe(env_df.head(30), use_container_width=True)

    if view_mode == "학교별 보기":
        school = st.sidebar.selectbox("학교 선택", schools)
        d = env_df[env_df["학교"] == school].copy()

        st.subheader(f"{school} - {metric} 그래프")

        if dt_col:
            d = d.sort_values(dt_col)
            fig = px.line(d, x=dt_col, y=metric, title=f"{school} / {metric} (시간 추세)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.histogram(d, x=metric, nbins=40, title=f"{school} / {metric} (분포)")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("요약 통계")
        st.dataframe(d[[metric]].describe().T, use_container_width=True)

    else:  # 학교 비교
        st.subheader(f"학교 비교 - {metric}")

        if dt_col:
            # 같은 시간축에서 비교가 가능하도록 일자 단위 집계(필요 시 변경 가능)
            env_df["__date__"] = pd.to_datetime(env_df[dt_col], errors="coerce").dt.date
            agg = (
                env_df.dropna(subset=["__date__"])
                .groupby(["학교", "__date__"], as_index=False)[metric]
                .mean()
            )
            fig = px.line(agg, x="__date__", y=metric, color="학교", title=f"{metric} (일자 평균) - 학교별 비교")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.box(env_df, x="학교", y=metric, points="outliers", title=f"{metric} - 학교별 분포 비교(Box)")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("학교별 요약(평균/표준편차/최솟값/최댓값)")
        summary = env_df.groupby("학교")[metric].agg(["count", "mean", "std", "min", "max"]).reset_index()
        st.dataframe(summary, use_container_width=True)

# -----------------------------
# Main: Growth XLSX
# -----------------------------
else:
    growth_df, xlsx_name = load_growth_xlsx()

    if growth_df.empty:
        if xlsx_name:
            st.warning(f"XLSX 파일({xlsx_name})은 찾았지만 읽기에 실패했습니다. 파일 형식/시트를 확인해 주세요.")
        else:
            st.warning("data/ 폴더에 .xlsx 파일을 찾지 못했습니다.")
        st.stop()

    st.subheader(f"생육결과 데이터: {xlsx_name}")
    st.dataframe(growth_df.head(50), use_container_width=True)

    # 학교 컬럼 추정 (있으면 필터 제공)
    possible_school_cols = [c for c in growth_df.columns if any(k in str(c) for k in ["학교", "고등학교", "school"])]
    school_col = possible_school_cols[0] if possible_school_cols else None

    if school_col:
        schools = sorted(growth_df[school_col].dropna().unique().tolist())
        sel = st.sidebar.selectbox("학교 선택", ["전체"] + schools)
        if sel != "전체":
            growth_df = growth_df[growth_df[school_col] == sel].copy()
            st.subheader(f"{sel} - 생육결과 미리보기")
            st.dataframe(growth_df.head(50), use_container_width=True)

    numeric_cols = growth_df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        metric = st.sidebar.selectbox("지표(수치 컬럼)", numeric_cols)
        fig = px.histogram(growth_df, x=metric, nbins=40, title=f"{metric} 분포")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("요약 통계")
        st.dataframe(growth_df[[metric]].describe().T, use_container_width=True)
    else:
        st.info("수치형 컬럼이 없어 그래프 대신 표만 표시합니다.")
