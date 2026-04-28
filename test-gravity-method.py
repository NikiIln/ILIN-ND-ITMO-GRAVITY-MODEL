#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
import folium
import os
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import seaborn as sns

def plot_gravity_heatmaps(df, gravity_features, x_col='x', y_col='y', grid_size=50):
    """
    Строит тепловые карты гравитационных потенциалов для каждой зоны.
    
    Параметры:
    - df: DataFrame с данными
    - gravity_features: DataFrame с гравитационными потенциалами (результат calculate_gravity_potential)
    - x_col, y_col: названия колонок с координатами
    - grid_size: размер сетки для интерполяции (больше = детальнее, но медленнее)
    """
    # Создаём общую фигуру с подграфиками для каждой зоны
    n_zones = len(gravity_features.columns)
    n_cols = 2  # 2 колонки на строку
    n_rows = (n_zones + n_cols - 1) // n_cols  # Расчёт строк
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 7 * n_rows))
    if n_zones == 1:
        axes = np.array([axes])  # Для единообразия обработки
    axes = axes.flatten()

    # Создаём сетку для интерполяции
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    for idx, zone_col in enumerate(gravity_features.columns):
        ax = axes[idx]

        # Интерполяция гравитационного потенциала на сетку
        from scipy.interpolate import griddata
        zi = griddata(
            (df[x_col], df[y_col]),
            df[zone_col],
            (Xi, Yi),
            method='linear'
        )

        # Построение тепловой карты
        im = ax.contourf(Xi, Yi, zi, levels=20, cmap='RdYlBu_r', alpha=0.7)
        ax.scatter(df[x_col], df[y_col], c='black', s=10, alpha=0.3)  # Исходные точки

        # Настройки графика
        ax.set_title(f'Гравитационный потенциал: {zone_col.replace("gravity_", "")}')
        ax.set_xlabel('X, м')
        ax.set_ylabel('Y, м')

        # Добавляем цветную шкалу
        plt.colorbar(im, ax=ax, label='Потенциал')

    # Убираем пустые подграфики
    for idx in range(n_zones, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

import folium
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

def create_gravity_folium_map(df, gravity_features, x_col='x', y_col='y', grid_size=50, zoom_start=12, output_path="gravity_potentials_interactive.html"):
    try:
        # Проверка данных
        if df.empty:
            print("Ошибка: DataFrame пуст")
            return None

        x_min, x_max = df[x_col].min(), df[x_col].max()
        y_min, y_max = df[y_col].min(), df[y_col].max()

        # Проверка на одинаковые координаты
        if x_min == x_max or y_min == y_max:
            print("Ошибка: все координаты x или y одинаковые. Невозможно создать карту.")
            return None

        center_lat = (y_min + y_max) / 2
        center_lon = (x_min + x_max) / 2

        # Создаём базовую карту
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

        # Создаём сетку для интерполяции
        xi = np.linspace(x_min, x_max, grid_size)
        yi = np.linspace(y_min, y_max, grid_size)
        Xi, Yi = np.meshgrid(xi, yi)

        for zone_col in gravity_features.columns:
            # Интерполяция гравитационного потенциала на сетку
            zi = griddata(
                (df[x_col], df[y_col]),
                df[zone_col],
                (Xi, Yi),
                method='linear'
            )

            # Проверка на все NaN
            if np.all(np.isnan(zi)):
                print(f"Пропускаем {zone_col}: все значения NaN после интерполяции")
                continue

            # Нормализация значений для лучшей визуализации
            zi_clean = np.nan_to_num(zi, nan=0.0)
            if np.max(zi_clean) != np.min(zi_clean):
                zi_normalized = (zi_clean - np.min(zi_clean)) / (np.max(zi_clean) - np.min(zi_clean))
            else:
                zi_normalized = zi_clean  # Если все значения одинаковые

            # Создаём слой тепловой карты
            folium.raster_layers.ImageOverlay(
                image=zi_normalized,
                bounds=[[y_min, x_min], [y_max, x_max]],
                colormap=plt.cm.RdYlBu_r,
                opacity=0.6,
                name=f'Потенциал: {zone_col.replace("gravity_", "")}'
            ).add_to(m)

        # Добавляем слой контроля для переключения между зонами
        folium.LayerControl().add_to(m)

        # Сохраняем карту
        full_path = os.path.abspath(output_path)
        m.save(full_path)
        print(f"Карта успешно сохранена: {full_path}")
        return m

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Добавляем слой контроля для переключения между зонами
    folium.LayerControl().add_to(m)
    return m



RNG = np.random.default_rng(42)

# ----------------------------
# 1) Синтетическая городская сцена
# ----------------------------
X0, X1, Y0, Y1 = 0.0, 10000.0, 0.0, 10000.0
CENTER = np.array([5000.0, 5000.0])               # городской центр
CBD = np.array([5400.0, 5200.0])                  # деловое ядро
IND_HUBS = np.array([[1700.0, 1800.0], [8200.0, 2100.0]])
PARKS = np.array([[2800.0, 7600.0], [6500.0, 7000.0], [4200.0, 3100.0]])
INST_HUBS = np.array([[3500.0, 5600.0], [7200.0, 6100.0]])
WATER_Y = 8200.0                                   # условный водоём — горизонтальная линия

ZONE_NAMES = np.array([
    "residential",
    "commercial",
    "industrial",
    "recreational",
    "institutional",
])

def min_dist(points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    return cKDTree(anchors).query(points, k=1)[0]

def dist_to_water(points: np.ndarray) -> np.ndarray:
    return np.abs(points[:, 1] - WATER_Y)

def slope_surface(points: np.ndarray) -> np.ndarray:
    # мягкий рельеф: сочетание градиента и волнообразной компоненты
    x = points[:, 0] / 1000.0
    y = points[:, 1] / 1000.0
    slope = 2.5 + 1.2 * np.sin(x / 1.5) + 1.0 * np.cos(y / 1.3) + 0.5 * (points[:, 0] / X1)
    return np.clip(slope, 0.2, 8.0)

def make_synthetic_blocks(n: int = 600) -> pd.DataFrame:
    hubs = np.array([
        [4200.0, 5400.0],  # ближе к центру
        [5600.0, 4700.0],  # ближе к CBD
        [1800.0, 2200.0],  # промышленный узел
        [7000.0, 7600.0],  # зелёно-водная зона
        [7200.0, 6100.0],  # институциональный кластер
        [3100.0, 6800.0],  # смешанный северо-запад
    ])
    labels = RNG.integers(0, len(hubs), size=n)
    xy = hubs[labels] + RNG.normal(0, 950.0, size=(n, 2))
    xy[:, 0] = np.clip(xy[:, 0], X0 + 100, X1 - 100)
    xy[:, 1] = np.clip(xy[:, 1], Y0 + 100, Y1 - 100)

    d_center = np.linalg.norm(xy - CENTER, axis=1)
    d_cbd = np.linalg.norm(xy - CBD, axis=1)
    d_ind = min_dist(xy, IND_HUBS)
    d_park = min_dist(xy, PARKS)
    d_inst = min_dist(xy, INST_HUBS)
    d_water = dist_to_water(xy)
    slope = slope_surface(xy)

    # осмысленные факторы
    green_index = np.clip(
        1.2 * np.exp(-d_park / 1400.0) + 0.9 * np.exp(-d_water / 1800.0) + RNG.normal(0, 0.07, n),
        0.0, 1.0
    )
    transport_access = np.clip(
        1.1 * np.exp(-d_cbd / 2600.0) + 0.8 * np.exp(-d_ind / 2300.0) + RNG.normal(0, 0.06, n),
        0.0, 1.5
    )
    land_value = np.clip(
        1.5 * np.exp(-d_center / 2400.0) + 0.8 * np.exp(-d_water / 2600.0) + 0.2 * green_index
        - 0.08 * slope + RNG.normal(0, 0.05, n),
        0.0, 2.0
    )
    jobs_access = np.clip(
        1.3 * np.exp(-d_cbd / 2200.0) + 0.9 * np.exp(-d_inst / 2600.0) + RNG.normal(0, 0.06, n),
        0.0, 2.0
)
    services_access = np.clip(
        1.1 * np.exp(-d_center / 2800.0) + 0.7 * np.exp(-d_inst / 1800.0) + 0.2 * green_index
        + RNG.normal(0, 0.05, n),
        0.0, 2.0
    )
    pop_density = np.clip(
        2500 + 8500 * np.exp(-d_center / 2500.0) + 1800 * services_access - 600 * slope + RNG.normal(0, 500, n),
        300, 18000
    )

    df = pd.DataFrame({
        "x": xy[:, 0],
        "y": xy[:, 1],
        "dist_center_m": d_center,
        "dist_cbd_m": d_cbd,
        "dist_industrial_m": d_ind,
        "dist_park_m": d_park,
        "dist_institutional_m": d_inst,
        "dist_water_m": d_water,
        "slope_deg": slope,
        "green_index": green_index,
        "transport_access": transport_access,
        "land_value": land_value,
        "jobs_access": jobs_access,
        "services_access": services_access,
        "pop_density": pop_density,
    })

    # ----------------------------
    # 2) Истинные классы: генеративная логика 5 зон
    # ----------------------------
    s_res = (
        0.30 * services_access + 0.20 * green_index + 0.18 * np.exp(-d_center / 2600.0)
        + 0.12 * np.exp(-d_inst / 2200.0) - 0.08 * slope + RNG.normal(0, 0.05, n)
    )
    s_com = (
        0.42 * jobs_access + 0.30 * np.exp(-d_cbd / 1700.0) + 0.18 * land_value
        + 0.08 * transport_access + RNG.normal(0, 0.05, n)
    )
    s_ind = (
        0.34 * transport_access + 0.25 * np.exp(-d_ind / 1800.0) + 0.16 * (1.2 - land_value)
        + 0.10 * (d_center / X1) - 0.08 * green_index + RNG.normal(0, 0.05, n)
    )
    s_rec = (
        0.38 * green_index + 0.24 * np.exp(-d_water / 1500.0) + 0.18 * np.exp(-d_park / 1400.0)
        - 0.10 * pop_density / 18000.0 - 0.05 * slope + RNG.normal(0, 0.05, n)
    )
    s_inst = (
        0.30 * np.exp(-d_inst / 1700.0) + 0.24 * services_access + 0.20 * jobs_access
        + 0.08 * land_value + 0.05 * green_index + RNG.normal(0, 0.05, n)
    )

    scores = np.vstack([s_res, s_com, s_ind, s_rec, s_inst]).T
    z = np.argmax(scores, axis=1)
    df["zone_true"] = ZONE_NAMES[z]
    return df

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.tight_layout()
    plt.show()

# ----------------------------
def calculate_gravity_potential(df, zone_col='zone_true', x_col='x', y_col='y'):
    """
    Расчёт гравитационного потенциала для каждого блока.
    """
    zones = df[zone_col].unique()
    gravity_features = {}

    for zone in zones:
        zone_data = df[df[zone_col] == zone]
        # Центр зоны (центр масс)
        center_x = zone_data[x_col].mean()
        center_y = zone_data[y_col].mean()
        # «Масса» зоны (количество объектов)
        mass = len(zone_data)

        # Расчёт расстояний от всех точек до центра зоны
        distances = np.sqrt(
            (df[x_col] - center_x)**2 + (df[y_col] - center_y)**2
        )
        # Гравитационный потенциал: масса / расстояние² (с регуляризацией)
        potential = mass / (distances**2 + 1e-8)
        gravity_features[f'gravity_{zone}'] = potential

    return pd.DataFrame(gravity_features)
# 3) CLUE-подобные гравитационные потенциалы
# ----------------------------
def clue_gravity_features(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "transport_access", "land_value", "jobs_access",
        "services_access", "green_index", "pop_density",
        "dist_center_m", "dist_cbd_m", "dist_industrial_m",
        "dist_park_m", "dist_institutional_m", "dist_water_m", "slope_deg"
    ]
    X = df[base_cols].copy()

    # приводим к логике "чем больше, тем лучше"
    X["dist_center_m"] *= -1
    X["dist_cbd_m"] *= -1
    X["dist_industrial_m"] *= -1
    X["dist_park_m"] *= -1
    X["dist_institutional_m"] *= -1
    X["dist_water_m"] *= -1
    X["slope_deg"] *= -1

    Z = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)

    # зоны задаются разными "массами" факторов
    mass = pd.DataFrame({
        "residential": (
            0.26 * Z["services_access"] + 0.18 * Z["green_index"] +
            0.16 * Z["pop_density"] + 0.14 * Z["dist_center_m"] +
            0.12 * Z["dist_institutional_m"] + 0.14 * Z["slope_deg"]
        ),
        "commercial": (
            0.30 * Z["jobs_access"] + 0.26 * Z["dist_cbd_m"] +
            0.18 * Z["land_value"] + 0.14 * Z["transport_access"] +
            0.12 * Z["dist_center_m"]
        ),
        "industrial": (
            0.28 * Z["transport_access"] + 0.24 * Z["dist_industrial_m"] +
            0.18 * (1 - Z["land_value"]) + 0.12 * (1 - Z["green_index"]) +
            0.10 * Z["dist_center_m"] + 0.08 * (1 - Z["dist_water_m"])
        ),
        "recreational": (
            0.30 * Z["green_index"] + 0.24 * Z["dist_park_m"] +
            0.20 * Z["dist_water_m"] + 0.14 * Z["slope_deg"] +
            0.12 * (1 - Z["pop_density"])
        ),
        "institutional": (
            0.28 * Z["dist_institutional_m"] + 0.22 * Z["services_access"] +
            0.20 * Z["jobs_access"] + 0.12 * Z["land_value"] +
            0.10 * Z["dist_center_m"] + 0.08 * Z["green_index"]
        ),
    })

    xy = df[["x", "y"]].to_numpy()
    tree = cKDTree(xy)
    beta = 1.8
    eps = 30.0
    radius = 2200.0

    out = pd.DataFrame(index=df.index)
    for zone in mass.columns:
        phi = np.zeros(len(df))
        m = mass[zone].to_numpy()
        for i in range(len(df)):
            idx = [j for j in tree.query_ball_point(xy[i], radius) if j != i]
            if not idx:
                continue
            d = np.linalg.norm(xy[idx] - xy[i], axis=1)
            phi[i] = np.sum(m[idx] / (d ** beta + eps))
        out[f"phi_{zone}"] = phi

    # переводим потенциалы в псевдо-вероятности CLUE
    phi_cols = [c for c in out.columns if c.startswith("phi_")]
    probs = out[phi_cols].copy()
    probs = probs.div(probs.sum(axis=1).replace(0, 1), axis=0)
    probs.columns = [c.replace("phi_", "p_clue_") for c in probs.columns]
    return pd.concat([mass.add_prefix("mass_"), out, probs], axis=1)

def spatial_groups(x: np.ndarray, n_splits: int = 5) -> np.ndarray:
    order = np.argsort(x)
    groups = np.empty(len(x), dtype=int)
    groups[order] = np.arange(len(x)) // max(1, len(x) // n_splits)
    return np.clip(groups, 0, n_splits - 1)

def main() -> None:
    df = make_synthetic_blocks(600)
    gravity = clue_gravity_features(df)
    df = pd.concat([df, gravity], axis=1)

    feature_cols = [
        "dist_center_m", "dist_cbd_m", "dist_industrial_m", "dist_park_m",
        "dist_institutional_m", "dist_water_m", "slope_deg",
        "green_index", "transport_access", "land_value",
        "jobs_access", "services_access", "pop_density",
        "p_clue_residential", "p_clue_commercial", "p_clue_industrial",
        "p_clue_recreational", "p_clue_institutional",
    ]

    X = df[feature_cols].to_numpy()
    y, classes = pd.factorize(df["zone_true"])
    groups = spatial_groups(df["x"].to_numpy(), n_splits=5)
    gkf = GroupKFold(n_splits=5)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        n_estimators=180,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=2,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=1,
    )

    y_oof = cross_val_predict(model, X, y, cv=gkf, groups=groups, method="predict")
    df["zone_pred_oof"] = classes[y_oof]

    acc = (df["zone_true"] == df["zone_pred_oof"]).mean()
    print(f"Spatial OOF accuracy: {acc:.3f}")
    print(classification_report(df["zone_true"], df["zone_pred_oof"], zero_division=0))
    print(confusion_matrix(df["zone_true"], df["zone_pred_oof"], labels=list(classes)))
if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import seaborn as sns




import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors  # Добавляем правильный импорт
import numpy as np

def plot_zoning_map(df, title="Функциональное зонирование"):
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # --- Левая панель: истинное зонирование ---
    unique_true = sorted(df['zone_true'].unique())
    n_classes = len(unique_true)

    # Создаём палитру с нужным количеством цветов
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    custom_cmap = mcolors.ListedColormap(colors)  # Исправлено: mcolors вместо plt.colors

    scatter1 = ax[0].scatter(
        df['x'], df['y'],
        c=df['zone_true'].map({cls: i for i, cls in enumerate(unique_true)}),
        cmap=custom_cmap,
        s=50
    )
    ax[0].set_title('Истинное зонирование')
    ax[0].set_xlabel('X, м')
    ax[0].set_ylabel('Y, м')

    handles1 = []
    labels1 = []
    for i, cls in enumerate(unique_true):
        marker = Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=colors[i], markersize=10)
        handles1.append(marker)
        labels1.append(cls)
    legend1 = ax[0].legend(handles=handles1, labels=labels1, title="Зоны")
    ax[0].add_artist(legend1)

    # --- Правая панель: предсказанное зонирование ---
    if 'zone_pred_oof' in df.columns and not df['zone_pred_oof'].isna().all():
        unique_pred = sorted(df['zone_pred_oof'].unique())
        n_pred_classes = len(unique_pred)
        pred_colors = plt.cm.tab10(np.linspace(0, 1, n_pred_classes))
        pred_cmap = mcolors.ListedColormap(pred_colors)  # Исправлено: mcolors вместо plt.colors

        scatter2 = ax[1].scatter(
            df['x'], df['y'],
            c=df['zone_pred_oof'].map({cls: i for i, cls in enumerate(unique_pred)}),
            cmap=pred_cmap,
            s=50,
            alpha=0.8
        )
        ax[1].set_title('Предсказанное зонирование')
        ax[1].set_xlabel('X, м')
        ax[1].set_ylabel('Y, м')

        handles2 = []
        labels2 = []
        for i, cls in enumerate(unique_pred):
            marker = Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=pred_colors[i], markersize=10)
            handles2.append(marker)
            labels2.append(cls)
        legend2 = ax[1].legend(handles=handles2, labels=labels2, title="Зоны")
        ax[1].add_artist(legend2)
    else:
        # Улучшенное сообщение об отсутствии предсказаний
        ax[1].clear()
        ax[1].text(0.5, 0.6, 'Предсказания отсутствуют',
                  transform=ax[1].transAxes, ha='center', va='center',
                  fontsize=14, fontweight='bold', color='red')
        ax[1].text(0.5, 0.4, 'Добавьте колонку "zone_pred_oof" в DataFrame',
                  transform=ax[1].transAxes, ha='center', va='center',
                  fontsize=10, color='gray')
        ax[1].set_title('Предсказания не рассчитаны')
        ax[1].set_xlabel('X, м')
        ax[1].set_ylabel('Y, м')
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()


# ----------------------------
# Основной блок выполнения
# ----------------------------

# 1. Генерируем синтетические данные
df = make_synthetic_blocks(n=600)

# 2. (Опционально) Добавляем предсказанные классы, если модель уже обучена
# df["zone_pred_oof"] = model.predict(X)  # замените X на ваши признаки


# 3. Проверяем, что df создан и содержит нужные колонки
print("Колонки DataFrame:", df.columns.tolist())
print("Первые 5 строк:")
print(df.head())

# 4. Визуализируем зонирование
plot_zoning_map(df)

# 5. (Опционально) Визуализируем спутниковые данные
# plot_satellite_features(df)  # если функция определена

# ----------------------------
# Основной блок выполнения
# ----------------------------

# ----------------------------
# Основной блок выполнения
# ----------------------------

# 1. Генерируем синтетические данные
df = make_synthetic_blocks(n=600)

# 2. Диагностика данных
print("=== ДИАГНОСТИКА ДАННЫХ ===")
print("Колонки DataFrame:", df.columns.tolist())
print("Уникальные классы в zone_true:", df['zone_true'].unique())

# 3. Расчёт гравитационных потенциалов
gravity_features = calculate_gravity_potential(df)
df = pd.concat([df, gravity_features], axis=1)

# 3.1. Визуализация гравитационных полей
plot_gravity_heatmaps(df, gravity_features)

# 3.2. Визуализация на интерактивной карте (folium)
create_gravity_folium_map(df, gravity_features)  # Вызов новой функции

# 4. Подготовка признаков и целевой переменной
feature_cols = ['x', 'y'] + [col for col in gravity_features.columns]
X = df[feature_cols]
y = df['zone_true']

# 4.1. Кодирование меток классов в числа
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 5. Инициализация модели XGBoost
model = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

# 6. Обучение и предсказание
model.fit(X, y_encoded)
y_pred_encoded = cross_val_predict(model, X, y_encoded, cv=5)

# 6.1. Декодирование предсказаний обратно в текст
df["zone_pred_oof"] = label_encoder.inverse_transform(y_pred_encoded)

# 7. Визуализация зонирования
plot_zoning_map(df)


# 8. Строим матрицу ошибок
classes = sorted(df['zone_true'].unique())
cm = confusion_matrix(df["zone_true"], df["zone_pred_oof"], labels=classes)
plot_confusion_matrix(cm, classes)  # Теперь функция определена


# 9. Важность признаков
importance = model.feature_importances_
feature_names = feature_cols
indices = np.argsort(importance)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Важность признаков в модели XGBoost (CLUE+гравитация)")
plt.bar(range(len(importance)), importance[indices])
plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

# 10. Анализ ошибок
df['is_correct'] = (df['zone_true'] == df['zone_pred_oof']).astype(int)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
df.groupby('zone_true')['is_correct'].mean().plot.bar(ax=ax[0])
ax[0].set_title('Доля верных предсказаний по классам')
ax[0].set_ylabel('Accuracy')
df['zone_true'].value_counts().plot.bar(ax=ax[1])
ax[1].set_title('Распределение классов в данных')
ax[1].set_ylabel('Количество блоков')
plt.tight_layout()
plt.show()

