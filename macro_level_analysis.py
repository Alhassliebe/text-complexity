import os
import logging
import re
import itertools
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, adjusted_rand_score
import umap.umap_ as umap


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_metrics_data(metrics_dir, segment_sizes=[10, 100]):
    """
    Загрузка данных метрик из CSV-файлов для всех размеров сегментов.
    """
    metrics_data = {size: {} for size in segment_sizes}
    
    logger.info(f"Поиск файлов метрик в директории {metrics_dir}")

    if not os.path.exists(metrics_dir):
        logger.error(f"Директория {metrics_dir} не существует")
        return metrics_data

    text_dirs = [d for d in os.listdir(metrics_dir) 
                if os.path.isdir(os.path.join(metrics_dir, d))]
    
    if not text_dirs:
        logger.warning(f"В директории {metrics_dir} не найдены поддиректории с текстами")
        return metrics_data
    
    logger.info(f"Найдено {len(text_dirs)} текстовых директорий")
    
    # Загружаем метрики для каждого текста и размера сегмента
    for text_name in tqdm(text_dirs, desc="Загрузка данных метрик"):
        text_dir = os.path.join(metrics_dir, text_name)
        
        for size in segment_sizes:
            metrics_file = os.path.join(text_dir, f"metrics_segment_{size}.csv")
            
            if os.path.exists(metrics_file):
                try:
                    df = pd.read_csv(metrics_file)
                    if not df.empty:
                        metrics_data[size][text_name] = df
                    else:
                        logger.warning(f"Пустой файл метрик: {metrics_file}")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке файла {metrics_file}: {e}")
            else:
                logger.debug(f"Файл метрик не найден: {metrics_file}")

    for size in segment_sizes:
        num_texts = len(metrics_data[size])
        logger.info(f"Загружены метрики для {num_texts} текстов (сегменты по {size} предложений)")
    
    return metrics_data


def extract_author_metadata(text_names):
    """
    Извлечение метаданных об авторе из названий текстов.
    """
    metadata = {}
    
    # Паттерны для определения авторов
    author_patterns = {
        r'^[Aa]bgaryan': 'Абгарян',
        r'^[Aa]kunin': 'Акунин',
        r'^[Aa]leshkovsky': 'Алешковский',
        r'^[Aa]stafiev': 'Астафьев',
        r'^[Bb]elyaev': 'Беляев',
        r'^[Bb]onchOsmolovskaya': 'Бонч-Осмоловская',
        r'^[Bb]ushkov': 'Бушков',
        r'^[Bb]ykov': 'Быков',
        r'^[Cc]herkasova': 'Черкасова',
        r'^[Dd]ombrowski': 'Домбровский',
        r'^[Dd]ostoevsky': 'Достоевский',
        r'^[Dd]ovlatov': 'Довлатов',
        r'^[Ee]xler': 'Экслер',
        r'^[Gg]elb': 'Гельб',
        r'^[Gg]ranin': 'Гранин',
        r'^[Gg]rekova': 'Грекова',
        r'^[Hh]offman': 'Хоффман',
        r'^[Ll]ermontov': 'Лермонтов',
        r'^[Ll]ukyanenko': 'Лукьяненко',
        r'^[Mm]akushinskiyi': 'Макушинский',
        r'^[Mm]arinina': 'Маринина',
        r'^[Mm]oshkov': 'Мошков',
        r'^[Pp]reobrazhensky': 'Преображенский',
        r'^[Pp]rokofeva': 'Прокофьева',
        r'^[Pp]ushkin': 'Пушкин',
        r'^[Rr]egentov': 'Регентов',
        r'^[Ss]alnikov': 'Сальников',
        r'^[Ss]hushpanov': 'Шушпанов',
        r'^[Ss]tepnova': 'Степнова',
        r'^[Ss]trugatskie': 'Стругацкие',
        r'^[Tt]olstoy': 'Толстой',
        r'^[Vv]lasova': 'Власова',
        r'^[Vv]oinovich': 'Войнович'
    }
    
    # Обработка каждого текста
    for text_name in text_names:
        metadata[text_name] = {}
        
        # Определение автора
        for pattern, author in author_patterns.items():
            if re.search(pattern, text_name, re.IGNORECASE):
                metadata[text_name]['author'] = author
                break
        else:
            metadata[text_name]['author'] = 'Неизвестный'

    return pd.DataFrame.from_dict(metadata, orient='index')


def aggregate_metrics(metrics_data, segment_size):
    """
    Агрегация метрик для каждого текста с вычислением статистик.
    """
    aggregated_metrics = {}
    
    # Список ключевых метрик из микроанализа
    key_metrics = [
        'avg_sentence_length', 'tree_depth', 'dependency_distance', 
        'subordinate_clauses_per_sentence', 'coordinating_constructions_per_sentence',
        'verbs_per_sentence', 'adjectives_per_sentence', 'nouns_per_sentence',
        'modifiers_per_noun', 'lexical_diversity', 'hapax_legomena_ratio',
        'avg_word_length', 'avg_syllables_per_word', 'morphological_complexity',
        'passive_voice_ratio', 'personal_pronouns_ratio', 'punctuation_per_sentence',
        'discourse_markers_per_sentence', 'lexical_density', 'flesch_index'
    ]
    
    for text_name, df in metrics_data.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        exclude_cols = ['segment_id', 'segment_size']
        metric_cols = [col for col in numeric_cols if col not in exclude_cols]

        available_metrics = [col for col in metric_cols if col in key_metrics]
        
        if not available_metrics:
            logger.warning(f"Для текста {text_name} не найдены нужные метрики. Пропускаем.")
            continue

        agg_metrics = {}

        agg_metrics['segment_size'] = segment_size

        for col in available_metrics:
            agg_metrics[f'{col}_mean'] = df[col].mean()
            agg_metrics[f'{col}_std'] = df[col].std()
            agg_metrics[f'{col}_median'] = df[col].median()
            agg_metrics[f'{col}_q25'] = df[col].quantile(0.25)
            agg_metrics[f'{col}_q75'] = df[col].quantile(0.75)
            agg_metrics[f'{col}_iqr'] = df[col].quantile(0.75) - df[col].quantile(0.25)

            if df[col].mean() != 0:
                agg_metrics[f'{col}_cv'] = df[col].std() / df[col].mean()
            else:
                agg_metrics[f'{col}_cv'] = 0

        aggregated_metrics[text_name] = agg_metrics

    if not aggregated_metrics:
        logger.warning(f"Не удалось агрегировать метрики ни для одного текста (сегменты по {segment_size})")
        return pd.DataFrame()

    agg_df = pd.DataFrame.from_dict(aggregated_metrics, orient='index')
    logger.info(f"Созданы агрегированные метрики для {len(agg_df)} текстов (сегменты по {segment_size})")

    return agg_df


def preprocess_data(data_df):
    """
    Предварительная обработка данных перед снижением размерности.
    """
    if data_df.empty:
        logger.error("Пустой DataFrame. Невозможно выполнить предобработку.")
        return np.array([]), []

    text_names = data_df.index.tolist()

    if 'segment_size' in data_df.columns:
        data_df = data_df.drop(columns=['segment_size'])

    constant_cols = [col for col in data_df.columns if data_df[col].nunique() <= 1]
    if constant_cols:
        logger.info(f"Удалено {len(constant_cols)} колонок-констант")
        data_df = data_df.drop(columns=constant_cols)

    if data_df.isnull().sum().sum() > 0:
        logger.warning(f"Найдено {data_df.isnull().sum().sum()} пропущенных значений. Заполняем средними.")
        data_df = data_df.fillna(data_df.mean())

    if data_df.empty or data_df.shape[1] == 0:
        logger.error("После предобработки не осталось данных.")
        return np.array([]), text_names

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_df)
    
    logger.info(f"Данные предобработаны: {scaled_data.shape[0]} текстов, {scaled_data.shape[1]} признаков")
    
    return scaled_data, text_names


def find_optimal_clusters(data, max_clusters=10, random_state=42):
    """
    Определение оптимального числа кластеров с помощью силуэтного анализа.
    """
    n_samples = data.shape[0]
    
    if n_samples < 3:
        logger.warning(f"Слишком мало данных для кластерного анализа (n_samples={n_samples}). Возвращаем 2 кластера.")
        return min(2, n_samples)

    max_possible_clusters = min(max_clusters, n_samples - 1)
    k_range = range(2, max_possible_clusters + 1)
    
    silhouette_scores = []
    
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(data)

            cluster_sizes = np.bincount(labels)
            min_cluster_size = cluster_sizes.min()
            
            if min_cluster_size < 1:
                logger.warning(f"Для k={k} есть пустые кластеры. Пропускаем.")
                silhouette_scores.append(-1)
                continue

            if k > 1 and min_cluster_size > 1:
                score = silhouette_score(data, labels)
                silhouette_scores.append(score)
                logger.info(f"Силуэтный коэффициент для k={k}: {score:.4f}")
            else:
                silhouette_scores.append(-1)
                logger.warning(f"Для k={k} невозможно вычислить силуэт (слишком мало данных в кластерах)")
        except Exception as e:
            logger.error(f"Ошибка при k={k}: {e}")
            silhouette_scores.append(-1)

    valid_scores = [(k, score) for k, score in zip(k_range, silhouette_scores) if score != -1]
    
    if not valid_scores:
        logger.warning("Не удалось вычислить силуэтные коэффициенты. Используем 2 кластера.")
        optimal_k = min(2, n_samples)
    else:
        optimal_k, max_score = max(valid_scores, key=lambda x: x[1])
        logger.info(f"Оптимальное число кластеров: {optimal_k} (силуэтный коэффициент: {max_score:.4f})")

    if valid_scores:
        k_values, score_values = zip(*valid_scores)
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, score_values, 'o-', color='blue')
        plt.axvline(x=optimal_k, color='red', linestyle='--')
        plt.title('Силуэтный анализ для определения оптимального числа кластеров', fontsize=14)
        plt.xlabel('Число кластеров', fontsize=12)
        plt.ylabel('Средний силуэтный коэффициент', fontsize=12)
        plt.grid(True, alpha=0.3)

        os.makedirs('output/visualizations/macro', exist_ok=True)
        plt.savefig('output/visualizations/macro/silhouette_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return optimal_k


def dimension_reduction(data, methods=['tsne', 'umap'], n_components=2, random_state=42):
    """
    Снижение размерности данных различными методами.
    """
    results = {}
    n_samples = data.shape[0]
    
    if n_samples < 2:
        logger.warning(f"Недостаточно данных для снижения размерности (n_samples={n_samples})")
        return results

    perplexity = min(30, max(2, n_samples // 4))
    n_neighbors = min(15, max(2, n_samples // 3))
    
    # t-SNE
    if 'tsne' in methods:
        try:
            logger.info(f"Применение t-SNE (perplexity={perplexity})")
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                init='pca',
                learning_rate='auto'
            )
            tsne_result = tsne.fit_transform(data)
            results['tsne'] = tsne_result
            logger.info("t-SNE успешно применен")
        except Exception as e:
            logger.error(f"Ошибка при применении t-SNE: {e}")
            # Запасной вариант - PCA
            from sklearn.decomposition import PCA
            logger.info("Использование PCA вместо t-SNE")
            pca = PCA(n_components=n_components, random_state=random_state)
            results['tsne'] = pca.fit_transform(data)
    
    # UMAP
    if 'umap' in methods:
        try:
            logger.info(f"Применение UMAP (n_neighbors={n_neighbors})")
            umap_reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric='euclidean',
                random_state=random_state
            )
            umap_result = umap_reducer.fit_transform(data)
            results['umap'] = umap_result
            logger.info("UMAP успешно применен")
        except Exception as e:
            logger.error(f"Ошибка при применении UMAP: {e}")
            # Запасной вариант - PCA
            if 'tsne' not in results:
                from sklearn.decomposition import PCA
                logger.info("Использование PCA вместо UMAP")
                pca = PCA(n_components=n_components, random_state=random_state)
                results['umap'] = pca.fit_transform(data)
            else:
                # Если t-SNE работал, используем его результаты для UMAP тоже
                logger.info("Использование результатов t-SNE вместо UMAP")
                results['umap'] = results['tsne']
    
    return results


def cluster_data(data, method='kmeans', n_clusters=None, random_state=42):
    """
    Кластеризация данных различными методами.
    """
    n_samples = data.shape[0]
    
    if n_samples < 2:
        logger.warning(f"Недостаточно данных для кластеризации (n_samples={n_samples})")
        return np.zeros(n_samples, dtype=int)

    if n_clusters is None:
        n_clusters = find_optimal_clusters(data, random_state=random_state)

    n_clusters = min(n_clusters, n_samples)
    
    # K-Means
    if method == 'kmeans':
        try:
            logger.info(f"Кластеризация K-means (n_clusters={n_clusters})")
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(data)
            logger.info(f"K-means: распределение по кластерам: {np.bincount(labels)}")
            return labels
        except Exception as e:
            logger.error(f"Ошибка при выполнении K-means: {e}")
            return np.zeros(n_samples, dtype=int)
    
    # Иерархическая кластеризация
    elif method == 'hierarchical':
        try:
            logger.info(f"Иерархическая кластеризация (n_clusters={n_clusters}, linkage='ward')")
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            labels = hierarchical.fit_predict(data)
            logger.info(f"Иерархическая: распределение по кластерам: {np.bincount(labels)}")
            return labels
        except Exception as e:
            logger.error(f"Ошибка при выполнении иерархической кластеризации: {e}")
            return np.zeros(n_samples, dtype=int)
    else:
        logger.error(f"Неподдерживаемый метод кластеризации: {method}")
        return np.zeros(n_samples, dtype=int)


def analyze_clusters(labels, metadata, group_by='author'):
    """
    Анализ состава кластеров и оценка качества кластеризации.
    """
    results = {
        'homogeneity': 0,
        'completeness': 0,
        'adjusted_rand_index': 0,
        'cluster_composition': {}
    }

    if group_by not in metadata.columns:
        logger.warning(f"Невозможно провести анализ по признаку '{group_by}': отсутствует в метаданных")
        return results

    unique_groups = metadata[group_by].nunique()
    if unique_groups < 2:
        logger.warning(f"В метаданных {group_by} меньше 2 уникальных значений ({unique_groups}). "
                      "Невозможно оценить качество кластеризации.")
        return results
    
    try:
        # Вычисление метрик оценки качества кластеризации
        homogeneity = homogeneity_score(metadata[group_by], labels)
        completeness = completeness_score(metadata[group_by], labels)

        unique_clusters = len(np.unique(labels))
        if unique_clusters < 2:
            logger.warning("Меньше 2 кластеров, ARI не может быть вычислен корректно.")
            adj_rand = 0
        else:
            adj_rand = adjusted_rand_score(metadata[group_by], labels)
        
        results['homogeneity'] = homogeneity
        results['completeness'] = completeness
        results['adjusted_rand_index'] = adj_rand
        
        logger.info(f"Оценка кластеризации по {group_by}:")
        logger.info(f"  Гомогенность: {homogeneity:.4f} - насколько каждый кластер содержит элементы одного класса")
        logger.info(f"  Полнота: {completeness:.4f} - насколько все элементы одного класса находятся в одном кластере")
        logger.info(f"  Скорректированный индекс Рэнда: {adj_rand:.4f} - сходство разбиений")
    except Exception as e:
        logger.error(f"Ошибка при оценке кластеризации по {group_by}: {e}")
    
    # Анализ состава кластеров
    try:
        cluster_composition = {}
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_size = mask.sum()
            
            if cluster_size == 0:
                logger.warning(f"Пустой кластер {cluster_id}. Пропускаем.")
                continue

            group_counts = Counter(metadata.loc[mask, group_by])

            if group_counts:
                dominant_group = max(group_counts.items(), key=lambda x: x[1])
                dominant_percentage = 100 * dominant_group[1] / cluster_size
                
                cluster_composition[cluster_id] = {
                    'size': cluster_size,
                    'dominant_group': dominant_group[0],
                    'dominant_percentage': dominant_percentage,
                    'composition': {group: count/cluster_size for group, count in group_counts.items()}
                }
                
                logger.info(f"Кластер {cluster_id} (размер: {cluster_size}):")
                logger.info(f"  Доминирующая группа: {dominant_group[0]} ({dominant_percentage:.1f}%)")
                for group, pct in sorted(cluster_composition[cluster_id]['composition'].items(), 
                                        key=lambda x: x[1], reverse=True):
                    logger.info(f"  {group}: {pct*100:.1f}%")

                texts_in_cluster = metadata.index[mask].tolist()
                cluster_composition[cluster_id]['texts'] = texts_in_cluster

        results['cluster_composition'] = cluster_composition
    except Exception as e:
        logger.error(f"Ошибка при анализе состава кластеров по {group_by}: {e}")
    return results


def visualize_clusters(data_2d, labels, metadata, text_names, 
                      output_dir="output/visualizations/macro", 
                      filename_prefix="clusters", 
                      title_prefix="Кластеризация текстов", 
                      reduction_method="t-SNE", 
                      clustering_method="K-means", 
                      figsize=(14, 12)):
    """
    Визуализация результатов кластеризации.
    """
    os.makedirs(output_dir, exist_ok=True)

    df_viz = pd.DataFrame(data_2d, columns=['x', 'y'])
    df_viz['text'] = text_names
    df_viz['cluster'] = labels

    for col in metadata.columns:
        if col in metadata.columns:
            df_viz[col] = metadata[col].values

    unique_labels = np.unique(labels)
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=figsize)
    
    for i, label in enumerate(unique_labels):
        mask = df_viz['cluster'] == label
        plt.scatter(df_viz.loc[mask, 'x'], df_viz.loc[mask, 'y'], 
                  color=cluster_colors[i], s=100, alpha=0.7, label=f"Кластер {label}")

    for i, row in df_viz.iterrows():
        plt.annotate(row['text'], (row['x'], row['y']), 
                    fontsize=8, alpha=0.7, 
                    xytext=(5, 5), textcoords='offset points')

    plt.title(f"{title_prefix}\n({reduction_method} + {clustering_method})", fontsize=16)
    plt.xlabel(f"{reduction_method} компонент 1", fontsize=12)
    plt.ylabel(f"{reduction_method} компонент 2", fontsize=12)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout()

    output_file = os.path.join(output_dir, f"{filename_prefix}_{reduction_method}_{clustering_method}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Визуализация кластеров сохранена в {output_file}")

    # Визуализация по автору
    category = 'author'
    if category in df_viz.columns:
        plt.figure(figsize=figsize)

        unique_categories = df_viz[category].unique()
        if len(unique_categories) <= 10:
            category_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
        elif len(unique_categories) <= 20:
            category_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))
        else:
            base_colors = [
                '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
                '#800000', '#008000', '#000080', '#808000', '#800080', '#008080',
                '#FFA500', '#FF69B4', '#32CD32', '#8A2BE2', '#DC143C', '#00CED1',
                '#FFD700', '#ADFF2F', '#FF1493', '#00BFFF', '#FF4500', '#9ACD32',
                '#FF6347', '#40E0D0', '#DA70D6', '#EE82EE', '#F0E68C', '#DDA0DD',
                '#B22222', '#228B22', '#4169E1', '#D2691E', '#CD5C5C', '#20B2AA',
                '#F4A460', '#6495ED', '#DEB887', '#5F9EA0', '#D3D3D3', '#A9A9A9'
            ]

            while len(base_colors) < len(unique_categories):

                import random
                random.seed(len(base_colors))
                r = random.choice([0, 128, 255])
                g = random.choice([0, 128, 255]) 
                b = random.choice([0, 128, 255])
                if r == g == b:
                    continue
                base_colors.append(f'#{r:02x}{g:02x}{b:02x}')

            import random
            random.seed(42)
            random.shuffle(base_colors)
            
            category_colors = base_colors[:len(unique_categories)]

        for i, cat_value in enumerate(unique_categories):
            mask = df_viz[category] == cat_value
            plt.scatter(df_viz.loc[mask, 'x'], df_viz.loc[mask, 'y'], 
                      color=category_colors[i], s=100, alpha=0.7, label=cat_value)

            for cluster_id in unique_labels:
                cluster_mask = mask & (df_viz['cluster'] == cluster_id)
                if cluster_mask.sum() > 0:
                    plt.scatter(df_viz.loc[cluster_mask, 'x'], df_viz.loc[cluster_mask, 'y'], 
                              edgecolor='black', s=120, facecolors='none', linewidth=1.5,
                              alpha=0.5)

        for i, row in df_viz.iterrows():
            plt.annotate(row['text'], (row['x'], row['y']), 
                        fontsize=8, alpha=0.7, 
                        xytext=(5, 5), textcoords='offset points')

        plt.title(f"{reduction_method}: Распределение текстов по {category}\n(метод кластеризации: {clustering_method})", 
                 fontsize=16)
        plt.xlabel(f"{reduction_method} компонент 1", fontsize=12)
        plt.ylabel(f"{reduction_method} компонент 2", fontsize=12)

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout()

        output_file = os.path.join(output_dir, f"{filename_prefix}_{reduction_method}_by_{category}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Визуализация по {category} сохранена в {output_file}")


def create_cluster_report(labels_dict, metadata, clustering_results, 
                         output_dir="output/reports"):
    """
    Создание отчета о результатах кластеризации в формате Markdown.
    """
    os.makedirs(output_dir, exist_ok=True)

    report_lines = [
        "# Отчет о результатах кластеризации текстов\n",
        "\n## Общая информация\n",
        f"- Количество текстов: {len(metadata)}\n",
    ]

    # Информация об авторах
    if 'author' in metadata.columns:
        values = sorted(metadata['author'].unique())
        report_lines.append(f"- Author: {', '.join(values)}\n")

    for method, labels in labels_dict.items():
        unique_clusters = len(np.unique(labels))
        report_lines.append(f"- Количество кластеров ({method}): {unique_clusters}\n")
    
    # Результаты кластеризации по методам
    for method, labels in labels_dict.items():
        report_lines.append(f"\n## Результаты метода {method}\n")
        
        group_by = 'author'
        if group_by in clustering_results and method in clustering_results[group_by]:
            results = clustering_results[group_by][method]
            
            report_lines.append(f"\n### Оценка кластеризации по {group_by}\n")
            report_lines.append(f"- Гомогенность: {results['homogeneity']:.4f}\n")
            report_lines.append(f"- Полнота: {results['completeness']:.4f}\n")
            report_lines.append(f"- Скорректированный индекс Рэнда: {results['adjusted_rand_index']:.4f}\n")

            if 'cluster_composition' in results:
                report_lines.append(f"\n### Состав кластеров по {group_by} ({method})\n")
                
                for cluster_id, info in results['cluster_composition'].items():
                    report_lines.append(f"\n#### Кластер {cluster_id}\n")
                    report_lines.append(f"- Размер: {info['size']} текстов\n")
                    report_lines.append(f"- Доминирующая группа: {info['dominant_group']} ({info['dominant_percentage']:.1f}%)\n")
                    report_lines.append("- Состав:\n")
                    
                    for group, pct in sorted(info['composition'].items(), key=lambda x: x[1], reverse=True):
                        report_lines.append(f"  - {group}: {pct*100:.1f}%\n")

                    if 'texts' in info:
                        report_lines.append("- Тексты в кластере:\n")
                        for text in info['texts']:
                            if text in metadata.index:
                                author = metadata.loc[text, 'author'] if 'author' in metadata.columns else 'Неизвестный'
                                report_lines.append(f"  - {text} ({author})\n")
            else:
                report_lines.append(f"\n### Состав кластеров по {group_by} не найден\n")

    report_path = os.path.join(output_dir, "clustering_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    logger.info(f"Отчет о кластеризации сохранен в {report_path}")


def compare_segment_sizes(results_10, results_100, output_dir="output/reports"):
    """
    Сравнение результатов кластеризации для разных размеров сегментов.
    """
    os.makedirs(output_dir, exist_ok=True)

    report_lines = [
        "# Сравнение результатов для разных размеров сегментов\n",
        "\n## Качество кластеризации\n",
    ]

    methods = set()
    for group_results in itertools.chain(results_10.values(), results_100.values()):
        methods.update(group_results.keys())

    for method in methods:
        report_lines.append(f"\n### Метод: {method}\n")
        
        group_by = 'author'
        if (group_by in results_10 and method in results_10[group_by] and
            group_by in results_100 and method in results_100[group_by]):
            
            report_lines.append(f"\n#### Группировка по {group_by}\n")
            report_lines.append("| Метрика | Сегменты по 10 предложений | Сегменты по 100 предложений | Разница |\n")
            report_lines.append("| --- | --- | --- | --- |\n")
            
            metrics = ['homogeneity', 'completeness', 'adjusted_rand_index']
            metric_names = {
                'homogeneity': 'Гомогенность',
                'completeness': 'Полнота',
                'adjusted_rand_index': 'Скорректированный индекс Рэнда'
            }
            
            for metric in metrics:
                val_10 = results_10[group_by][method][metric]
                val_100 = results_100[group_by][method][metric]
                diff = val_100 - val_10
                diff_sign = "+" if diff > 0 else ""
                
                report_lines.append(f"| {metric_names[metric]} | {val_10:.4f} | {val_100:.4f} | {diff_sign}{diff:.4f} |\n")

    report_lines.append("\n## Общее заключение\n")

    better_size = {'10': 0, '100': 0, 'равно': 0}
    
    group_by = 'author'
    if group_by in results_10 and group_by in results_100:
        for method in methods:
            if method in results_10[group_by] and method in results_100[group_by]:
                for metric in ['homogeneity', 'completeness', 'adjusted_rand_index']:
                    val_10 = results_10[group_by][method][metric]
                    val_100 = results_100[group_by][method][metric]
                    
                    if abs(val_10 - val_100) < 0.001:
                        better_size['равно'] += 1
                    elif val_10 > val_100:
                        better_size['10'] += 1
                    else:
                        better_size['100'] += 1

    if better_size['10'] > better_size['100']:
        report_lines.append("По большинству метрик сегменты размером 10 предложений дают лучшие результаты кластеризации. ")
        report_lines.append(f"(Метрики лучше для сегментов по 10 предложений: {better_size['10']}, ")
        report_lines.append(f"для сегментов по 100 предложений: {better_size['100']}, равны: {better_size['равно']})\n")
    elif better_size['100'] > better_size['10']:
        report_lines.append("По большинству метрик сегменты размером 100 предложений дают лучшие результаты кластеризации. ")
        report_lines.append(f"(Метрики лучше для сегментов по 100 предложений: {better_size['100']}, ")
        report_lines.append(f"для сегментов по 10 предложений: {better_size['10']}, равны: {better_size['равно']})\n")
    else:
        report_lines.append("Оба размера сегментов (10 и 100 предложений) дают сопоставимые результаты кластеризации. ")
        report_lines.append(f"(Метрики лучше для сегментов по 10 предложений: {better_size['10']}, ")
        report_lines.append(f"для сегментов по 100 предложений: {better_size['100']}, равны: {better_size['равно']})\n")
    
    # Выводы о влиянии размера сегмента на кластеризацию
    report_lines.append("\n### Выводы о влиянии размера сегмента на кластеризацию\n")
    
    report_lines.append("1. **Детализация анализа**: ")
    if better_size['10'] > better_size['100']:
        report_lines.append("Сегменты по 10 предложений позволяют обнаруживать более тонкие различия в стиле ")
        report_lines.append("и обеспечивают более детальный анализ, что может быть полезно для выявления особенностей ")
        report_lines.append("авторского стиля на микроуровне.\n")
    elif better_size['100'] > better_size['10']:
        report_lines.append("Сегменты по 100 предложений лучше отражают общий стиль текста, ")
        report_lines.append("что позволяет точнее определять авторство на макроуровне.\n")
    else:
        report_lines.append("Оба размера сегментов могут иметь свои преимущества в зависимости от задачи: ")
        report_lines.append("сегменты по 10 предложений - для анализа микроуровня, а по 100 предложений - для макроуровня.\n")
    
    report_lines.append("\n2. **Стабильность метрик**: ")
    report_lines.append("Стабильность метрик между разными размерами сегментов указывает на устойчивость ")
    report_lines.append("авторского стиля и возможность его надежной идентификации.\n")
    
    report_lines.append("\n3. **Рекомендации для будущих исследований**: ")
    report_lines.append("Для комплексного анализа рекомендуется использовать оба размера сегментов, ")
    report_lines.append("что позволит получить как детальный, так и обобщенный взгляд на стилистические особенности текстов.\n")

    report_path = os.path.join(output_dir, "segment_size_comparison.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    logger.info(f"Отчет сравнения размеров сегментов сохранен в {report_path}")


def main():
    """
    Основная функция для выполнения макроанализа текстов.
    """
    metrics_dir = "output/metrics"
    output_dir = "output/visualizations/macro"
    reports_dir = "output/reports"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    logger.info("Начало макроанализа текстов")
    
    # 1. Загрузка данных метрик для разных размеров сегментов
    logger.info("Загрузка данных метрик")
    metrics_data = load_metrics_data(metrics_dir, segment_sizes=[10, 100])

    if not any(metrics_data.values()):
        logger.error("Не найдены данные метрик. Убедитесь, что микроанализ был выполнен корректно.")
        return

    segment_results = {}
    
    # 2. Обработка для каждого размера сегмента
    for segment_size in [10, 100]:
        segment_dir = os.path.join(output_dir, f"segment_{segment_size}")
        os.makedirs(segment_dir, exist_ok=True)
        
        logger.info(f"\n=== Обработка сегментов по {segment_size} предложений ===")

        if not metrics_data[segment_size]:
            logger.warning(f"Нет данных для сегментов по {segment_size} предложений. Пропускаем.")
            continue

        if len(metrics_data[segment_size]) < 2:
            logger.warning(f"Недостаточно текстов для анализа (найдено {len(metrics_data[segment_size])}). "
                          "Требуется минимум 2 текста.")
            continue
        
        try:
            # 3. Агрегация метрик
            logger.info("Агрегация метрик")
            aggregated_df = aggregate_metrics(metrics_data[segment_size], segment_size)
            
            if aggregated_df.empty:
                logger.warning(f"Не удалось агрегировать метрики для сегментов по {segment_size} предложений. Пропускаем.")
                continue
            
            # 4. Извлечение метаданных
            logger.info("Извлечение метаданных из названий текстов")
            metadata_df = extract_author_metadata(aggregated_df.index)
            
            # 5. Предварительная обработка данных
            logger.info("Предварительная обработка данных")
            preprocessed_data, text_names = preprocess_data(aggregated_df)
            
            if len(preprocessed_data) == 0 or len(text_names) == 0:
                logger.warning("После предобработки не осталось данных. Пропускаем анализ.")
                continue
            
            # 6. Снижение размерности
            logger.info("Снижение размерности")
            reduced_data = dimension_reduction(preprocessed_data, methods=['tsne', 'umap'])
            
            if not reduced_data:
                logger.warning("Не удалось выполнить снижение размерности. Пропускаем анализ.")
                continue
            
            # 7. Кластеризация
            logger.info("Определение оптимального числа кластеров")
            optimal_k = find_optimal_clusters(preprocessed_data) # число кластеров 2
            # optimal_k = 4
            
            # K-means кластеризация
            logger.info(f"Кластеризация K-means (k={optimal_k})")
            kmeans_labels = cluster_data(preprocessed_data, method='kmeans', n_clusters=optimal_k)
            
            # Иерархическая кластеризация
            logger.info(f"Иерархическая кластеризация (k={optimal_k})")
            hierarchical_labels = cluster_data(preprocessed_data, method='hierarchical', n_clusters=optimal_k)
            
            # 8. Анализ результатов кластеризации
            logger.info("Анализ результатов кластеризации")
            clustering_results = {}

            group_by = 'author'
            if group_by in metadata_df.columns:
                logger.info(f"Анализ кластеризации по {group_by}")
                
                kmeans_analysis = analyze_clusters(kmeans_labels, metadata_df, group_by)
                hierarchical_analysis = analyze_clusters(hierarchical_labels, metadata_df, group_by)
                
                clustering_results[group_by] = {
                    'kmeans': kmeans_analysis,
                    'hierarchical': hierarchical_analysis
                }
            
            # 9. Визуализация результатов
            logger.info("Визуализация результатов")

            for reduction_method, reduced in reduced_data.items():
                visualize_clusters(
                    reduced, kmeans_labels, metadata_df, text_names,
                    output_dir=segment_dir,
                    filename_prefix=f"kmeans",
                    title_prefix=f"K-means кластеризация (сегменты по {segment_size} предложений)",
                    reduction_method=reduction_method.upper(),
                    clustering_method="K-means"
                )
                
                # Иерархические кластеры
                visualize_clusters(
                    reduced, hierarchical_labels, metadata_df, text_names,
                    output_dir=segment_dir,
                    filename_prefix=f"hierarchical",
                    title_prefix=f"Иерархическая кластеризация (сегменты по {segment_size} предложений)",
                    reduction_method=reduction_method.upper(),
                    clustering_method="Иерархическая"
                )
            
            # 10. Создание отчета о кластеризации
            logger.info("Создание отчета о кластеризации")
            segment_reports_dir = os.path.join(reports_dir, f"segment_{segment_size}")
            os.makedirs(segment_reports_dir, exist_ok=True)
            
            create_cluster_report(
                {'kmeans': kmeans_labels, 'hierarchical': hierarchical_labels},
                metadata_df,
                clustering_results,
                output_dir=segment_reports_dir
            )
            segment_results[segment_size] = clustering_results
            
            # 11. Сохранение результатов в CSV
            results_df = pd.DataFrame({
                'text_name': text_names,
                'kmeans_cluster': kmeans_labels,
                'hierarchical_cluster': hierarchical_labels
            })

            for column in metadata_df.columns:
                results_df[column] = metadata_df.loc[text_names, column].values

            results_file = os.path.join(segment_reports_dir, "clustering_results.csv")
            results_df.to_csv(results_file, index=False)
            logger.info(f"Результаты кластеризации сохранены в {results_file}")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке сегментов размером {segment_size}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 12. Сравнение результатов для разных размеров сегментов
    if 10 in segment_results and 100 in segment_results:
        logger.info("Сравнение результатов для разных размеров сегментов")
        compare_segment_sizes(segment_results[10], segment_results[100], output_dir=reports_dir)
    else:
        logger.warning("Невозможно сравнить результаты для разных размеров сегментов: недостаточно данных")
    
    logger.info("Макроанализ завершен!")


if __name__ == "__main__":
    main()