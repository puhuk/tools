def _test_distrib_integration_multilabel(device):

    rank = idist.get_rank()
    
    def _test(n_epochs, metric_device):
        metric_device = torch.device(metric_device)
        n_iters = 3
        s = 5
        n_classes = 2

        torch.manual_seed(12 + rank)

        offset = n_iters * s
        y_true = torch.randint(0, 2, size=(offset, n_classes, 8, 10)).to(device)
        y_preds = torch.randint(0, 2, size=(offset, n_classes, 8, 10)).to(device)

        def update(engine, i):
            return (
                y_preds[i * s : (i + 1) * s, ...],
                y_true[i * s : (i + 1) * s, ...],
            )

        engine = Engine(update)

        # Seems error occurs
        y_true = idist.all_reduce(y_true)
        y_preds = idist.all_reduce(y_preds)

        acc = Accuracy(is_multilabel=True, device=metric_device)
        acc.attach(engine, "acc")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert (
            acc._num_correct.device == metric_device
        ), f"{type(acc._num_correct.device)}:{acc._num_correct.device} vs {type(metric_device)}:{metric_device}"

        assert "acc" in engine.state.metrics
        res = engine.state.metrics["acc"]
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        true_res = accuracy_score(to_numpy_multilabel(y_true), to_numpy_multilabel(y_preds))

        assert pytest.approx(res) == true_res

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        for _ in range(2):
            _test(n_epochs=1, metric_device=metric_device)
            _test(n_epochs=2, metric_device=metric_device)
