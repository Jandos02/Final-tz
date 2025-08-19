import requests
import json

# URL API
BASE_URL = "http://localhost:8000"


def test_health():
    """Тестирование health check"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)


def test_model_info():
    """Тестирование получения информации о модели"""
    response = requests.get(f"{BASE_URL}/model_info")
    print("Model Info:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print("-" * 50)


def test_prediction():
    """Тестирование прогнозирования"""

    # Тестовые данные клиентов
    test_clients = [
        {
            "name": "Клиент с низким риском",
            "data": {
                "кредитный_рейтинг": 800,
                "город": "Алматы",
                "пол": "Female",
                "возраст": 25,
                "стаж_в_банке": 3,
                "баланс_депозита": 150000,
                "число_продуктов": 2,
                "есть_кредитка": 1,
                "активный_клиент": 1,
                "оценочная_зарплата": 100000,
            },
        },
        {
            "name": "Клиент с высоким риском",
            "data": {
                "кредитный_рейтинг": 500,
                "город": "Атырау",
                "пол": "Male",
                "возраст": 55,
                "стаж_в_банке": 1,
                "баланс_депозита": 0,
                "число_продуктов": 1,
                "есть_кредитка": 0,
                "активный_клиент": 0,
                "оценочная_зарплата": 50000,
            },
        },
        {
            "name": "Клиент без баланса депозита",
            "data": {
                "кредитный_рейтинг": 650,
                "город": "Астана",
                "пол": "Male",
                "возраст": 35,
                "стаж_в_банке": 5,
                "число_продуктов": 2,
                "есть_кредитка": 1,
                "активный_клиент": 1,
                "оценочная_зарплата": 120000,
            },
        },
    ]

    for client in test_clients:
        print(f"Тестирование: {client['name']}")
        response = requests.post(
            f"{BASE_URL}/predict",
            json=client["data"],
            headers={"Content-Type": "application/json"},
        )

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Вероятность оттока: {result['score']:.3f}")
            print(f"Прогноз: {'Уйдет' if result['prediction'] else 'Останется'}")
        else:
            print(f"Ошибка: {response.text}")
        print("-" * 50)


def test_invalid_data():
    """Тестирование с некорректными данными"""
    print("Тестирование с некорректными данными:")

    invalid_data = {
        "кредитный_рейтинг": "invalid",  # должно быть число
        "город": "Неизвестный город",
        "пол": "Other",
        "возраст": -5,
        "стаж_в_банке": 100,
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=invalid_data,
        headers={"Content-Type": "application/json"},
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    print("-" * 50)


if __name__ == "__main__":
    print("Тестирование Bank Churn Prediction API")
    print("=" * 50)

    try:
        # Тестируем основные функции
        test_health()
        test_model_info()
        test_prediction()
        test_invalid_data()

        print("Тестирование завершено!")

    except requests.exceptions.ConnectionError:
        print("Ошибка подключения к API. Убедитесь, что сервис запущен на порту 8000")
    except Exception as e:
        print(f"Ошибка при тестировании: {e}")
