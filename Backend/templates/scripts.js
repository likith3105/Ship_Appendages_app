$(document).ready(function () {
    $('#train_resnet').click(function () {
        $.ajax({
            url: '/train/resnet',
            type: 'POST',
            success: function (response) {
                alert(response.message);
            },
            error: function (error) {
                alert(error.responseJSON.error);
            }
        });
    });

    $('#train_vggnet').click(function () {
        $.ajax({
            url: '/train/vggnet',
            type: 'POST',
            success: function (response) {
                alert(response.message);
            },
            error: function (error) {
                alert(error.responseJSON.error);
            }
        });
    });

    $('#train_cnn').click(function () {
        $.ajax({
            url: '/train/cnn',
            type: 'POST',
            success: function (response) {
                alert(response.message);
            },
            error: function (error) {
                alert(error.responseJSON.error);
            }
        });
    });

    $('#predict').click(function () {
        // Assuming you have data to send for prediction
        var data = {};
        $.ajax({
            url: '/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(data),
            success: function (response) {
                // Assuming you want to display the predictions
                console.log(response);
            },
            error: function (error) {
                console.log(error);
            }
        });
    });
});
