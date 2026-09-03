from src.pipeline.prediction_pipeline import CustomData


def test_custom_data_stores_plain_scalars_not_tuples():
    data = CustomData(
        Type_of_order="Snack",
        Type_of_vehicle="motorcycle",
        Festival="No",
        City="Urban",
        Delivery_city="RES",
        Road_traffic_density="Low",
        Weather_conditions="Sunny",
        Delivery_person_Age=25.0,
        Delivery_person_Ratings=4.5,
        Vehicle_condition=1.0,
        multiple_deliveries=0.0,
        Time_Orderd_hour=10.0,
        distance=5.0,
    )

    assert data.Type_of_order == "Snack"
    assert data.distance == 5.0


def test_custom_data_as_dataframe_has_expected_columns():
    data = CustomData(
        Type_of_order="Snack",
        Type_of_vehicle="motorcycle",
        Festival="No",
        City="Urban",
        Delivery_city="RES",
        Road_traffic_density="Low",
        Weather_conditions="Sunny",
        Delivery_person_Age=25.0,
        Delivery_person_Ratings=4.5,
        Vehicle_condition=1.0,
        multiple_deliveries=0.0,
        Time_Orderd_hour=10.0,
        distance=5.0,
    )

    df = data.get_data_as_dataframe()

    assert len(df) == 1
    assert df.loc[0, "distance"] == 5.0
    assert df.loc[0, "Festival"] == "No"
