import os
import logging
from collections import defaultdict, Counter

import conllu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from razdel import sentenize
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# инициализация Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

RU_VOWELS = set('аеёиоуыэюя')

# Список дискурсивных маркеров (взято из интернета, можно заменить на свои)
DISCOURSE_MARKERS = {
    'однако', 'тем не менее', 'следовательно', 'таким образом', 
    'с одной стороны', 'с другой стороны', 'кроме того', 'в результате', 
    'например', 'несмотря на', 'более того', 'наоборот', 'впрочем', 
    'в частности', 'итак', 'вследствие', 'затем', 'наконец', 'прежде всего'
}


def load_conllu_file(file_path):
    """Загрузка файла CoNLL-U."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        sentences = conllu.parse(content)
        return sentences
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла {file_path}: {e}")
        return None


def extract_sentences_from_conllu(sentences):
    """Извлечение предложений из структуры CoNLL-U."""
    extracted = []
    for sentence in sentences:
        text = ' '.join([token['form'] for token in sentence])
        extracted.append(text)
    return extracted


def segment_text_by_sentences(sentences, segment_size=10):
    """Сегментация списка предложений на фрагменты по количеству предложений."""
    segments = []
    for i in range(0, len(sentences), segment_size):
        segment = sentences[i:i+segment_size]
        segments.append(segment)
    
    return segments


def count_syllables_ru(word):
    """Подсчет количества слогов в русском слове."""
    return sum(1 for char in word.lower() if char in RU_VOWELS)


def calculate_tree_depth(sentence):
    """Вычисление глубины дерева зависимостей."""
    children = defaultdict(list)
    for token in sentence:
        if token['head'] != 0:
            children[token['head']].append(token['id'])

    def depth(node_id):
        if not children[node_id]:
            return 0
        return 1 + max(depth(child) for child in children[node_id])

    root_id = next(token['id'] for token in sentence if token['head'] == 0)
    return depth(root_id)


def calculate_dependency_distance(sentence):
    """Вычисление средней зависимостной дистанции."""
    distances = []
    for token in sentence:
        if token['head'] != 0:
            distances.append(abs(token['id'] - token['head']))
    
    return sum(distances) / len(distances) if distances else 0


def calculate_metrics(sentences_segment, conllu_segment):
    """Расчет метрик сложности для сегмента текста."""
    metrics = {}
    
    # Объединяем все предложения сегмента в один текст для некоторых метрик
    full_text = ' '.join(sentences_segment)
    
    # Токенизация текста
    tokens = [_.text for _ in segmenter.tokenize(full_text)]
    sentences = sentences_segment
    
    # Проверка на наличие букв в тексте вообще
    alpha_tokens = [t for t in tokens if t.isalpha()]
    if not alpha_tokens:
        # Вернем нулевые значения для всех метрик при отсутствии слов с буквами
        metrics = {
            'avg_sentence_length': 0, 'avg_word_length': 0,
            'avg_syllables_per_word': 0, 'lexical_diversity': 0,
            'hapax_legomena_ratio': 0, 'tree_depth': 0,
            'dependency_distance': 0, 'subordinate_clauses_per_sentence': 0,
            'coordinating_constructions_per_sentence': 0,
            'verbs_per_sentence': 0, 'adjectives_per_sentence': 0,
            'nouns_per_sentence': 0, 'modifiers_per_noun': 0,
            'passive_voice_ratio': 0, 'personal_pronouns_ratio': 0,
            'punctuation_per_sentence': 0, 'morphological_complexity': 0,
            'discourse_markers_per_sentence': 0, 'lexical_density': 0,
            'flesch_index': 0
        }
        return metrics
    
    # 1. Средняя длина предложения (ASL)
    words_per_sentence = [len([_.text for _ in segmenter.tokenize(sent)]) for sent in sentences]
    metrics['avg_sentence_length'] = sum(words_per_sentence) / len(sentences) if sentences else 0
    
    # 2. Средняя длина слова
    metrics['avg_word_length'] = sum(len(token) for token in tokens) / len(tokens) if tokens else 0
    
    # 3. Средняя длина слова в слогах
    syllables = [count_syllables_ru(token) for token in alpha_tokens]  # Используем alpha_tokens вместо проверки
    metrics['avg_syllables_per_word'] = sum(syllables) / len(syllables) if syllables else 0
    
    # 4. Лексическое разнообразие (TTR)
    unique_tokens = set(token.lower() for token in alpha_tokens)  # Используем alpha_tokens вместо проверки
    metrics['lexical_diversity'] = len(unique_tokens) / len(alpha_tokens) if alpha_tokens else 0
    
    # 5. Доля Hapax Legomena (слов, встречающихся в тексте только один раз)
    word_counts = Counter(token.lower() for token in alpha_tokens)  # Используем alpha_tokens
    hapax_count = sum(1 for word, count in word_counts.items() if count == 1)
    metrics['hapax_legomena_ratio'] = hapax_count / len(unique_tokens) if unique_tokens else 0
    
    # 6. Синтаксические метрики из CoNLL-U
    total_tree_depth = 0
    total_dep_distance = 0
    total_clauses = 0
    total_coordinating = 0
    total_verbs = 0
    total_adjectives = 0
    total_nouns = 0
    total_noun_modifiers = 0
    total_passive_verbs = 0
    total_personal_pronouns = 0
    total_punct = 0
    total_morphological_features = 0
    total_tokens_with_features = 0
    total_discourse_markers = 0
    
    for sentence in conllu_segment:
        # Глубина дерева зависимостей
        tree_depth = calculate_tree_depth(sentence)
        total_tree_depth += tree_depth
        
        # Средняя зависимостная дистанция
        dep_distance = calculate_dependency_distance(sentence)
        total_dep_distance += dep_distance
        
        # Количество клауз (подчинительных предложений)
        clause_deps = sum(1 for token in sentence if token['deprel'] in ['advcl', 'acl', 'ccomp', 'xcomp'])
        total_clauses += clause_deps
        
        # Количество сочинительных конструкций
        coord_deps = sum(1 for token in sentence if token['deprel'] in ['conj', 'cc'])
        total_coordinating += coord_deps
        
        # Части речи
        verbs = sum(1 for token in sentence if token['upostag'] == 'VERB')
        total_verbs += verbs
        
        adjectives = sum(1 for token in sentence if token['upostag'] == 'ADJ')
        total_adjectives += adjectives
        
        nouns = sum(1 for token in sentence if token['upostag'] == 'NOUN')
        total_nouns += nouns
        
        # Модификаторы для существительных
        for token in sentence:
            if token['upostag'] == 'NOUN':
                modifiers = sum(1 for t in sentence if t['head'] == token['id'] and t['deprel'] in ['amod', 'nmod', 'appos'])
                total_noun_modifiers += modifiers
        
        # Страдательный залог
        passive_verbs = sum(1 for token in sentence if token['upostag'] == 'VERB' and 
                            'feats' in token and token['feats'] and 
                            'Voice' in token['feats'] and token['feats']['Voice'] == 'Pass')
        total_passive_verbs += passive_verbs
        
        # Личные местоимения
        personal_pronouns = sum(1 for token in sentence if token['upostag'] == 'PRON' and 
                            'feats' in token and token['feats'] and 
                            'Person' in token['feats'])
        total_personal_pronouns += personal_pronouns
        
        # Знаки пунктуации
        punct = sum(1 for token in sentence if token['upostag'] == 'PUNCT')
        total_punct += punct
        
        # Морфологическая сложность
        for token in sentence:
            if 'feats' in token and token['feats']:
                total_morphological_features += len(token['feats'])
                total_tokens_with_features += 1
        
        # Дискурсивные маркеры
        sent_text = ' '.join([token['form'] for token in sentence]).lower()
        for marker in DISCOURSE_MARKERS:
            if marker in sent_text:
                total_discourse_markers += 1
    
    # Вычисляем средние значения для синтаксических метрик
    num_sentences = len(conllu_segment)
    metrics['tree_depth'] = total_tree_depth / num_sentences if num_sentences else 0
    metrics['dependency_distance'] = total_dep_distance / num_sentences if num_sentences else 0
    metrics['subordinate_clauses_per_sentence'] = total_clauses / num_sentences if num_sentences else 0
    metrics['coordinating_constructions_per_sentence'] = total_coordinating / num_sentences if num_sentences else 0
    metrics['verbs_per_sentence'] = total_verbs / num_sentences if num_sentences else 0
    metrics['adjectives_per_sentence'] = total_adjectives / num_sentences if num_sentences else 0
    metrics['nouns_per_sentence'] = total_nouns / num_sentences if num_sentences else 0
    
    # Среднее число модификаторов на имя
    metrics['modifiers_per_noun'] = total_noun_modifiers / total_nouns if total_nouns else 0
    
    # Доля страдательного залога
    metrics['passive_voice_ratio'] = total_passive_verbs / total_verbs if total_verbs else 0
    
    # Доля личных местоимений
    metrics['personal_pronouns_ratio'] = total_personal_pronouns / len(tokens) if tokens else 0
    
    # Знаки пунктуации на предложение
    metrics['punctuation_per_sentence'] = total_punct / num_sentences if num_sentences else 0
    
    # Морфологическая сложность
    metrics['morphological_complexity'] = total_morphological_features / total_tokens_with_features if total_tokens_with_features else 0
    
    # Дискурсивные маркеры на предложение
    metrics['discourse_markers_per_sentence'] = total_discourse_markers / num_sentences if num_sentences else 0
    
    # Лексическая плотность
    content_words = sum(1 for token in tokens if token.isalpha() and token.lower() not in {'и', 'в', 'с', 'на', 'к', 'у', 'по', 'за', 'из', 'под', 'над', 'о', 'об', 'от', 'до', 'для', 'при', 'а', 'но', 'что', 'как', 'или', 'если', 'то', 'не', 'ни', 'да', 'бы'})
    metrics['lexical_density'] = content_words / len([t for t in tokens if t.isalpha()]) if tokens else 0
    
    # Индекс Флеша
    asl = metrics['avg_sentence_length']
    asw = metrics['avg_syllables_per_word']
    metrics['flesch_index'] = 206.835 - (1.52 * asl) - (65.14 * asw)
    
    return metrics


def process_text_file(file_path, segment_sizes=[10, 100], output_dir="output/metrics"):
    """Обработка файла с расчетом метрик по сегментам и их сохранением."""
    text_name = os.path.basename(file_path).split('.')[0]
    
    text_metrics_dir = os.path.join(output_dir, text_name)
    os.makedirs(text_metrics_dir, exist_ok=True)

    conllu_data = load_conllu_file(file_path)
    if not conllu_data:
        logger.error(f"Не удалось загрузить файл {file_path}")
        return None, text_name
    
    sentences = extract_sentences_from_conllu(conllu_data)
    
    results = {}
    
    for size in segment_sizes:
        sentence_segments = segment_text_by_sentences(sentences, segment_size=size)
        conllu_segments = segment_text_by_sentences(conllu_data, segment_size=size)
        
        logger.info(f"Файл {os.path.basename(file_path)}: создано {len(sentence_segments)} сегментов по {size} предложений")
        
        metrics_list = []
        for i, (sent_segment, conllu_segment) in enumerate(zip(sentence_segments, conllu_segments)):
            segment_metrics = calculate_metrics(sent_segment, conllu_segment)
            segment_metrics['segment_id'] = i
            segment_metrics['segment_size'] = size
            metrics_list.append(segment_metrics)
        
        df_metrics = pd.DataFrame(metrics_list)
        results[size] = df_metrics
        
        output_file = os.path.join(text_metrics_dir, f"metrics_segment_{size}.csv")
        df_metrics.to_csv(output_file, index=False)
        logger.info(f"Метрики сохранены в {output_file}")
    
    return results, text_name


def visualize_metrics(text_name, metrics_data, output_dir, key_metrics=None):
    """Визуализация метрик для разных сегментов текста."""
    os.makedirs(output_dir, exist_ok=True)
    
    text_output_dir = os.path.join(output_dir, text_name)
    os.makedirs(text_output_dir, exist_ok=True)
    
    # Ключевые метрики, для которых строится визуализация
    if key_metrics is None:
        key_metrics = [
            # Синтаксические метрики
            'avg_sentence_length', 'tree_depth', 'dependency_distance', 
            'subordinate_clauses_per_sentence', 'coordinating_constructions_per_sentence',
            
            # Части речи и грамматические метрики
            'verbs_per_sentence', 'adjectives_per_sentence', 'nouns_per_sentence', 
            'modifiers_per_noun', 'passive_voice_ratio', 'personal_pronouns_ratio',
            'punctuation_per_sentence', 'morphological_complexity',
            
            # Лексические метрики
            'lexical_diversity', 'lexical_density', 'hapax_legomena_ratio', 
            'avg_word_length', 'avg_syllables_per_word',
            
            # Дискурсивные и сводные метрики
            'discourse_markers_per_sentence', 'flesch_index'
        ]
    
    metric_labels = {
        'avg_sentence_length': 'Количество слов в предложении',
        'lexical_diversity': 'Коэффициент лексического разнообразия',
        'tree_depth': 'Средняя глубина синтаксического дерева',
        'dependency_distance': 'Средняя зависимостная дистанция',
        'subordinate_clauses_per_sentence': 'Количество подчинительных предложений',
        'coordinating_constructions_per_sentence': 'Количество сочинительных конструкций',
        'verbs_per_sentence': 'Количество глаголов в предложении',
        'flesch_index': 'Индекс удобочитаемости Флеша',
        'morphological_complexity': 'Индекс морфологической сложности',
        'adjectives_per_sentence': 'Количество прилагательных в предложении',
        'nouns_per_sentence': 'Количество существительных в предложении',
        'modifiers_per_noun': 'Количество модификаторов на существительное',
        'avg_syllables_per_word': 'Среднее количество слогов в слове',
        'avg_word_length': 'Средняя длина слова',
        'passive_voice_ratio': 'Доля страдательного залога',
        'personal_pronouns_ratio': 'Доля личных местоимений',
        'punctuation_per_sentence': 'Количество знаков препинания в предложении',
        'lexical_density': 'Коэффициент лексической плотности',
        'hapax_legomena_ratio': 'Доля уникальных слов (Hapax Legomena)',
        'discourse_markers_per_sentence': 'Количество дискурсивных маркеров в предложении'
    }
    
    for segment_size, df in metrics_data.items():
        segment_dir = os.path.join(text_output_dir, f"segment_{segment_size}")
        os.makedirs(segment_dir, exist_ok=True)
        
        num_segments = len(df)
        
        available_metrics = [col for col in df.columns if col not in ['segment_id', 'segment_size']]
        filtered_metrics = [m for m in key_metrics if m in available_metrics]
        
        # Создание визуализации для метрик
        for metric in filtered_metrics:
            plt.figure(figsize=(14, 7))
            
            metric_label = metric_labels.get(metric, metric)
            
            plt.plot(df['segment_id'], df[metric], marker='o', markersize=3, 
                     linestyle='-', alpha=0.5, color='darkblue')
            
            window_size = max(3, num_segments // 20)
            rolling_avg = df[metric].rolling(window=window_size, center=True).mean()
            plt.plot(df['segment_id'], rolling_avg, color='red', linewidth=2, 
                     label=f'Скользящее среднее (окно={window_size})')
            
            if num_segments > 500:
                step = num_segments // 8
                key_positions = list(range(0, num_segments, step))
                if (num_segments - 1) not in key_positions:
                    key_positions.append(num_segments - 1)
            else:
                key_positions = [
                    0,
                    num_segments // 4,
                    num_segments // 2,
                    3 * num_segments // 4,
                    num_segments - 1
                ]
            
            plt.xticks(key_positions, [int(pos) for pos in key_positions], fontsize=9, rotation=0)
            plt.legend(loc='upper right')
            
            plt.title(f"{text_name} - {metric_label} (сегменты по {segment_size} предложений)")
            plt.xlabel("Номер сегмента")
            plt.ylabel(metric_label)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            output_file = os.path.join(segment_dir, f"{metric}.png")
            plt.savefig(output_file, dpi=150)
            plt.close()

        # Группировка метрик по категориям
        metric_groups = {
            'Синтаксическая сложность': {
                'metrics': ['avg_sentence_length', 'tree_depth', 'dependency_distance', 
                           'subordinate_clauses_per_sentence', 'coordinating_constructions_per_sentence'],
                'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
            },
            'Грамматико-лексическая насыщенность': {
                'metrics': ['verbs_per_sentence', 'adjectives_per_sentence', 'nouns_per_sentence',
                           'modifiers_per_noun', 'morphological_complexity'],
                'colors': ['#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E', '#E17055']
            },
            'Лексическое разнообразие и плотность': {
                'metrics': ['lexical_diversity', 'lexical_density', 'hapax_legomena_ratio',
                           'avg_word_length', 'avg_syllables_per_word'],
                'colors': ['#00B894', '#00CEC9', '#55A3FF', '#FD79A8', '#FDCB6E']
            },
            'Стилистические и дискурсивные особенности': {
                'metrics': ['discourse_markers_per_sentence', 'passive_voice_ratio', 
                           'personal_pronouns_ratio', 'punctuation_per_sentence', 'flesch_index'],
                'colors': ['#E84393', '#9B59B6', '#3498DB', '#F39C12', '#E74C3C']
            }
        }
        
        # Создание 4 отдельных графиков только для сегментов по 100 предложений
        if segment_size == 100:
            # Подготовка нормализованных данных
            df_norm = df.copy()
            all_available_metrics = []
            for group_data in metric_groups.values():
                for metric in group_data['metrics']:
                    if metric in filtered_metrics:
                        all_available_metrics.append(metric)
                        min_val = df_norm[metric].min()
                        max_val = df_norm[metric].max()
                        if max_val > min_val:
                            df_norm[metric] = (df_norm[metric] - min_val) / (max_val - min_val)
            
            # Создание сетки 2x2
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            axes = axes.flatten()
            
            for idx, (group_name, group_data) in enumerate(metric_groups.items()):
                if idx >= 4:
                    break
                    
                ax = axes[idx]
                
                # Фильтрация доступных метрик для этой группы
                group_metrics = [m for m in group_data['metrics'] if m in filtered_metrics]
                
                if not group_metrics:
                    ax.text(0.5, 0.5, f'Нет данных для группы\n"{group_name}"', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title(group_name, fontsize=14, pad=20, weight='normal')
                    continue
                
                # Определение окна сглаживания
                window_size = max(3, num_segments // 15)
                
                # Построение графиков для метрик группы
                for i, metric in enumerate(group_metrics):
                    metric_label = metric_labels.get(metric, metric)
                    color = group_data['colors'][i % len(group_data['colors'])]
                    
                    # Скользящее среднее для сглаживания
                    rolling_avg = df_norm[metric].rolling(window=window_size, center=True).mean()
                    ax.plot(df_norm['segment_id'], rolling_avg, color=color, linewidth=2.5, 
                           label=metric_label, alpha=0.8)
                
                # Настройка осей
                if num_segments > 500:
                    step = num_segments // 8
                    key_positions = list(range(0, num_segments, step))
                    if (num_segments - 1) not in key_positions:
                        key_positions.append(num_segments - 1)
                else:
                    key_positions = [0, num_segments // 4, num_segments // 2, 
                                   3 * num_segments // 4, num_segments - 1]
                
                ax.set_xticks(key_positions)
                ax.set_xticklabels([int(pos) for pos in key_positions], fontsize=12)
                ax.set_xlabel("Номер сегмента", fontsize=13)
                ax.set_ylabel("Нормализованное значение", fontsize=13)
                ax.set_title(group_name, fontsize=15, pad=15, weight='normal')
                
                # Легенда
                leg = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, framealpha=0.9, handlelength=1)
                ax.grid(True, linestyle='--', alpha=0.4)
                
                # Цветовая схема фона
                ax.set_facecolor('#FAFAFA')
            
            plt.suptitle(f"{text_name} - Анализ метрик по категориям\n"
                        f"(сегменты по {segment_size} предложений, окно сглаживания: {window_size})", 
                        fontsize=18, y=0.98, weight='bold')
            plt.tight_layout()
            plt.subplots_adjust(left=0.05, right=0.85, hspace=0.3, wspace=0.9)
            
            output_file = os.path.join(segment_dir, "metrics_by_categories.png")
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            plt.close()
            
            logger.info(f"График по категориям сохранен: {output_file}")
        
        # Общий график для всех метрик
        important_metrics = filtered_metrics

        df_norm = df.copy()
        for col in important_metrics:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)

        plt.figure(figsize=(16, 10))
        colors = plt.cm.tab20.colors

        for i, metric in enumerate(important_metrics):
            metric_label = metric_labels.get(metric, metric)
            color = colors[i % len(colors)]

            window_size = max(5, num_segments // 15) if segment_size == 10 else max(3, num_segments // 15)
            rolling_avg = df_norm[metric].rolling(window=window_size, center=True).mean()
            plt.plot(df_norm['segment_id'], rolling_avg, color=color, linewidth=2.5,
                     label=metric_label)

        if num_segments > 500:
            step = num_segments // 8
            key_positions = list(range(0, num_segments, step))
            if (num_segments - 1) not in key_positions:
                key_positions.append(num_segments - 1)
        else:
            key_positions = [
                0,
                num_segments // 4,
                num_segments // 2,
                3 * num_segments // 4,
                num_segments - 1
            ]

        plt.xticks(key_positions, [int(pos) for pos in key_positions], fontsize=9, rotation=0)
        plt.xlabel("Номер сегмента", fontsize=12)
        plt.ylabel("Нормализованное значение", fontsize=12)
        plt.title(f"{text_name} - Сравнение нормализованных метрик (сегменты по {segment_size} предложений)",
                  fontsize=14, pad=20)

        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.subplots_adjust(right=0.7)

        output_file = os.path.join(segment_dir, "all_metrics_comparison.png")
        plt.savefig(output_file, dpi=150)
        plt.close()

        # ключевые метрики для которых строим обобщенный график
        important_metrics = [
            'avg_word_length', 'lexical_density', 'lexical_diversity', 'morphological_complexity',
            'avg_sentence_length', 'dependency_distance', 'personal_pronouns_ratio', 'flesch_index'
        ]

        available_important_metrics = [m for m in important_metrics if m in available_metrics]

        df_norm = df.copy()
        for col in available_important_metrics:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)

        # Создаем линейный график для всех нормализованных метрик
        plt.figure(figsize=(16, 10))
        colors = plt.cm.tab10.colors

        # Тут строится только скользящие средние для каждой метрики (иначе очень много значений, ничего не понятно)
        for i, metric in enumerate(available_important_metrics):
            metric_label = metric_labels.get(metric, metric)
            color = colors[i % len(colors)]

            window_size = max(5, num_segments // 15) if segment_size == 10 else max(3, num_segments // 15)
            rolling_avg = df_norm[metric].rolling(window=window_size, center=True).mean()
            plt.plot(df_norm['segment_id'], rolling_avg, color=color, linewidth=2.5,
                     label=metric_label)

        if num_segments > 500:
            step = num_segments // 8
            key_positions = list(range(0, num_segments, step))
            if (num_segments - 1) not in key_positions:
                key_positions.append(num_segments - 1)
        else:
            key_positions = [
                0,
                num_segments // 4,
                num_segments // 2,
                3 * num_segments // 4,
                num_segments - 1
            ]

        plt.xticks(key_positions, [int(pos) for pos in key_positions], fontsize=9, rotation=0)
        plt.xlabel("Номер сегмента", fontsize=12)
        plt.ylabel("Нормализованное значение", fontsize=12)
        plt.title(
            f"{text_name} - Сравнение ключевых нормализованных метрик (сегменты по {segment_size} предложений)",
            fontsize=14, pad=20)
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.subplots_adjust(right=0.7)

        output_file = os.path.join(segment_dir, "key_metrics_comparison.png")
        plt.savefig(output_file, dpi=150)
        plt.close()


def main():
    corpus_dir = "data/Authors_conllu"
    output_dir = "output/visualizations"
    metrics_dir = "output/metrics"
    reports_dir = "output/reports"
    
    if not os.path.exists(corpus_dir):
        logger.error(f"Директория {corpus_dir} не найдена")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    conllu_files = []
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            if file.endswith('.conllu'):
                conllu_files.append(os.path.join(root, file))
    
    logger.info(f"Найдено {len(conllu_files)} CoNLL-U файлов")

    if not conllu_files:
        logger.error("Не найдено файлов CoNLL-U для обработки")
        return
    
    key_metrics = [
        # Синтаксические метрики
        'avg_sentence_length', 'tree_depth', 'dependency_distance', 
        'subordinate_clauses_per_sentence', 'coordinating_constructions_per_sentence',
        
        # Части речи и грамматические метрики
        'verbs_per_sentence', 'adjectives_per_sentence', 'nouns_per_sentence', 
        'modifiers_per_noun', 'passive_voice_ratio', 'personal_pronouns_ratio',
        'punctuation_per_sentence', 'morphological_complexity',
        
        # Лексические метрики
        'lexical_diversity', 'lexical_density', 'hapax_legomena_ratio', 
        'avg_word_length', 'avg_syllables_per_word',
        
        # Дискурсивные и сводные метрики
        'discourse_markers_per_sentence', 'flesch_index'
    ]

    # ----------------------------------------------------------------------------------------------
    # Можно запустить для одного текста, для проверки. Либо запустить для всех текстов из директории.
    # ----------------------------------------------------------------------------------------------

    # ВАЖНО №1: Обрабатываем только первый файл для демонстрации (РАСКОММЕНТИРОВАТЬ ПРИ НЕОХОДИМОСТИ)
    # file_path = conllu_files[3]
    # text_name = os.path.basename(file_path).split('.')[0]
    
    # logger.info(f"Обработка файла {text_name}")
    
    # Список ключевых метрик для визуализации

    
    # # Обработка текста и расчет метрик с сохранением
    # metrics_data, text_name = process_text_file(file_path, 
    #                                segment_sizes=[10, 100], 
    #                                output_dir=metrics_dir)
    
    # if metrics_data:
    #     # Визуализация метрик с использованием только ключевых метрик
    #     visualize_metrics(text_name, metrics_data, output_dir=output_dir, key_metrics=key_metrics)
    #     logger.info(f"Визуализации для {text_name} сохранены в директории {output_dir}")
    # else:
    #     logger.warning(f"Не удалось обработать файл {text_name}")
    
    # ВАЖНО №2: Обработка всех файлов в указанной директории. (ЗАКОММЕНТИРОВАТЬ ПРИ НЕОХОДИМОСТИ)
    # Сбор всех метрик для анализа
    all_metrics = {10: {}, 100: {}}

    for file_path in tqdm(conllu_files, desc="Обработка файлов"):
        text_name = os.path.basename(file_path).split('.')[0]
        
        logger.info(f"Обработка файла {text_name}")
        
        # Обработка текста и расчет метрик с сохранением
        metrics_data, text_name = process_text_file(file_path, 
                                       segment_sizes=[10, 100], 
                                       output_dir=metrics_dir)
        
        if metrics_data:
            # Сохраняем для дальнейшего анализа
            for size in [10, 100]:
                if size in metrics_data:
                    all_metrics[size][text_name] = metrics_data[size]
            
            # Стандартная визуализация
            visualize_metrics(text_name, metrics_data, output_dir=output_dir, key_metrics=key_metrics)
            logger.info(f"Визуализации для {text_name} сохранены в директории {output_dir}")
        else:
            logger.warning(f"Не удалось обработать файл {text_name}")
    
    # НОВЫЙ АНАЛИЗ: Анализ вариативности авторов и корпуса
    logger.info("\n=== Начало анализа вариативности авторов ===")
    
    for segment_size in [10, 100]:
        if all_metrics[segment_size]:
            logger.info(f"Анализ для сегментов по {segment_size} предложений")
            analyze_author_variability(all_metrics[segment_size], segment_size)
    
    logger.info("Анализ завершен! Проверьте папки output/author_analysis и output/corpus_analysis")


def create_author_swarmplot(metrics_data, segment_size, author_name, output_dir="output/author_analysis"):
    """
    Создание swarmplot для произведений одного автора.
    Показывает неоднородность автора по различным метрикам.
    """
    import re
    
    os.makedirs(output_dir, exist_ok=True)

    author_texts = {}
    author_pattern = f"^{re.escape(author_name)}"
    
    for text_name, df in metrics_data.items():
        if re.search(author_pattern, text_name, re.IGNORECASE):
            author_texts[text_name] = df
    
    if len(author_texts) < 2:
        logger.warning(f"Найдено менее 2 текстов для автора {author_name}. Swarmplot не создан.")
        return
    
    # Ключевые метрики для анализа
    key_metrics = [
        'avg_word_length', 'lexical_density', 'lexical_diversity', 'morphological_complexity',
        'avg_sentence_length', 'dependency_distance', 'personal_pronouns_ratio', 'flesch_index'
    ]

    author_data = []
    for text_name, df in author_texts.items():

        short_name = text_name.replace(f"{author_name}_", "").replace(f"{author_name}2_", "")
        text_summary = {'text': short_name, 'full_text': text_name}
        
        for metric in key_metrics:
            if metric in df.columns:
                text_summary[f'{metric}_mean'] = df[metric].mean()
                text_summary[f'{metric}_std'] = df[metric].std()
        
        author_data.append(text_summary)
    
    author_df = pd.DataFrame(author_data)

    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    metric_labels = {
        'avg_word_length': 'Длина слов',
        'lexical_density': 'Лексич. плотность',
        'lexical_diversity': 'Лексич. разнообразие',
        'morphological_complexity': 'Морф. сложность',
        'avg_sentence_length': 'Длина предложения',
        'dependency_distance': 'Зависим. дистанция',
        'personal_pronouns_ratio': 'Личные местоимения',
        'flesch_index': 'Индекс Флеша'
    }
    
    for i, metric in enumerate(key_metrics):
        if i < len(axes):
            mean_col = f'{metric}_mean'
            if mean_col in author_df.columns:
                ax = axes[i]

                sns.swarmplot(data=author_df, y=mean_col, ax=ax, size=8, alpha=0.7, color='steelblue')

                for j, (idx, row) in enumerate(author_df.iterrows()):
                    ax.annotate(row['text'], 
                              (0, row[mean_col]), 
                              xytext=(10, 0), 
                              textcoords='offset points',
                              fontsize=8, 
                              rotation=15,
                              alpha=0.8)

                mean_val = author_df[mean_col].mean()
                std_val = author_df[mean_col].std()
                
                ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, 
                          label=f'μ = {mean_val:.3f}')
                ax.axhline(y=mean_val + std_val, color='orange', linestyle=':', alpha=0.5,
                          label=f'σ = {std_val:.3f}')
                ax.axhline(y=mean_val - std_val, color='orange', linestyle=':', alpha=0.5)
                
                ax.set_title(metric_labels.get(metric, metric), fontsize=12)
                ax.set_ylabel('Значение', fontsize=10)
                ax.set_xlabel('')
                ax.legend(fontsize=8, loc='upper right')
                ax.grid(True, alpha=0.3)

                ax.set_xticks([])

    for i in range(len(key_metrics), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(f'Вариативность стиля автора {author_name}\n'
                 f'(сегменты по {segment_size} предложений, n={len(author_texts)} произведений)', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"{author_name}_swarmplot_segment_{segment_size}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Swarmplot для автора {author_name} сохранен в {output_file}")

    stats_data = []
    for metric in key_metrics:
        mean_col = f'{metric}_mean'
        if mean_col in author_df.columns:
            values = author_df[mean_col]
            cv = values.std() / values.mean() if values.mean() != 0 else 0
            stats_data.append({
                'Метрика': metric_labels.get(metric, metric),
                'Среднее': f"{values.mean():.3f}",
                'Стд.откл.': f"{values.std():.3f}",
                'Мин.': f"{values.min():.3f}",
                'Макс.': f"{values.max():.3f}",
                'Коэф.вар.': f"{cv:.3f}"
            })
    
    stats_df = pd.DataFrame(stats_data)
    stats_file = os.path.join(output_dir, f"{author_name}_stats_segment_{segment_size}.csv")
    stats_df.to_csv(stats_file, index=False, encoding='utf-8')
    
    return author_df


def create_corpus_distribution_analysis(all_metrics_data, segment_size, output_dir="output/corpus_analysis"):
    """
    Анализ распределения метрик по всему корпусу и сравнение с индивидуальной вариативностью.
    """
    os.makedirs(output_dir, exist_ok=True)

    corpus_data = []
    for text_name, df in all_metrics_data.items():

        author = text_name.split('_')[0] if '_' in text_name else text_name.split('2_')[0] if '2_' in text_name else 'Unknown'

        key_metrics = [
            'avg_word_length', 'lexical_density', 'lexical_diversity', 'morphological_complexity',
            'avg_sentence_length', 'dependency_distance', 'personal_pronouns_ratio', 'flesch_index'
        ]
        
        text_summary = {
            'text': text_name,
            'author': author
        }
        
        for metric in key_metrics:
            if metric in df.columns:
                text_summary[f'{metric}_mean'] = df[metric].mean()
                text_summary[f'{metric}_std'] = df[metric].std()
        
        corpus_data.append(text_summary)
    
    corpus_df = pd.DataFrame(corpus_data)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    metric_labels = {
        'avg_word_length': 'Длина слов',
        'lexical_density': 'Лексич. плотность',
        'lexical_diversity': 'Лексич. разнообразие',
        'morphological_complexity': 'Морф. сложность',
        'avg_sentence_length': 'Длина предложения',
        'dependency_distance': 'Зависим. дистанция',
        'personal_pronouns_ratio': 'Личные местоимения',
        'flesch_index': 'Индекс Флеша'
    }
    
    key_metrics = list(metric_labels.keys())
    
    for i, metric in enumerate(key_metrics):
        if i < len(axes):
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            
            if mean_col in corpus_df.columns:
                ax = axes[i]

                means = corpus_df[mean_col].dropna()
                ax.hist(means, bins=15, alpha=0.7, color='skyblue', edgecolor='black', 
                       label='Распределение средних')

                corpus_mean = means.mean()
                corpus_std = means.std()
                corpus_median = means.median()
                
                ax.axvline(corpus_mean, color='red', linestyle='--', linewidth=2, 
                          label=f'Среднее: {corpus_mean:.3f}')
                ax.axvline(corpus_median, color='green', linestyle='--', linewidth=2,
                          label=f'Медиана: {corpus_median:.3f}')
                ax.axvline(corpus_mean + corpus_std, color='orange', linestyle=':', alpha=0.7)
                ax.axvline(corpus_mean - corpus_std, color='orange', linestyle=':', alpha=0.7)

                if std_col in corpus_df.columns:
                    avg_within_std = corpus_df[std_col].mean()
                    ax.text(0.02, 0.98, f'Средняя σ внутри текстов: {avg_within_std:.3f}\n'
                                        f'σ между текстами: {corpus_std:.3f}', 
                           transform=ax.transAxes, fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_title(f'{metric_labels.get(metric, metric)}\n'
                           f'Корпус: μ={corpus_mean:.3f}, σ={corpus_std:.3f}', fontsize=11)
                ax.set_xlabel('Значение метрики', fontsize=10)
                ax.set_ylabel('Количество текстов', fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

    for i in range(len(key_metrics), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(f'Распределение метрик по корпусу vs внутритекстовая вариативность\n'
                 f'(сегменты по {segment_size} предложений, n={len(corpus_df)} текстов)', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"corpus_distribution_analysis_segment_{segment_size}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    summary_stats = []
    for metric in key_metrics:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        if mean_col in corpus_df.columns:
            means = corpus_df[mean_col].dropna()
            within_stds = corpus_df[std_col].dropna() if std_col in corpus_df.columns else []
            
            summary_stats.append({
                'Метрика': metric_labels.get(metric, metric),
                'Среднее_по_корпусу': f"{means.mean():.3f}",
                'СтдОткл_между_текстами': f"{means.std():.3f}",
                'Среднее_СтдОткл_внутри_текстов': f"{within_stds.mean():.3f}" if len(within_stds) > 0 else "N/A",
                'Отношение_между_к_внутри': f"{means.std() / within_stds.mean():.2f}" if len(within_stds) > 0 and within_stds.mean() > 0 else "N/A",
                'Минимум': f"{means.min():.3f}",
                'Максимум': f"{means.max():.3f}"
            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_file = os.path.join(output_dir, f"corpus_summary_segment_{segment_size}.csv")
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    
    logger.info(f"Анализ распределения корпуса сохранен в {output_dir}")
    
    return corpus_df


def find_authors_with_multiple_works(metrics_data):
    """
    Поиск авторов с несколькими произведениями для swarmplot анализа.
    """
    authors_count = defaultdict(list)
    
    for text_name in metrics_data.keys():

        if '_' in text_name:
            author = text_name.split('_')[0]
        elif '2_' in text_name:
            author = text_name.split('2_')[0]
        else:
            author = text_name
        
        authors_count[author].append(text_name)

    multi_work_authors = {author: texts for author, texts in authors_count.items() 
                         if len(texts) >= 3}

    logger.info(f"Авторы с множественными произведениями:")
    for author, texts in multi_work_authors.items():
        logger.info(f"  {author}: {len(texts)} произведений")
    
    return multi_work_authors


def analyze_author_variability(metrics_data, segment_size, output_dir="output/author_analysis"):
    """
    Основная функция для анализа вариативности авторов.
    """
    os.makedirs(output_dir, exist_ok=True)
    multi_work_authors = find_authors_with_multiple_works(metrics_data)
    
    if not multi_work_authors:
        logger.warning("Не найдено авторов с достаточным количеством произведений")
        return

    for author in multi_work_authors.keys():
        logger.info(f"Создание swarmplot для автора {author}")
        create_author_swarmplot(metrics_data, segment_size, author, output_dir)

    logger.info("Создание анализа распределения по корпусу")
    create_corpus_distribution_analysis(metrics_data, segment_size, 
                                       output_dir.replace("author_analysis", "corpus_analysis"))


if __name__ == "__main__":
    main()