import torch
import logging
import torch.nn.functional as F
import util.utils as utils


def _train_or_test(model, epoch, dataloader, tb_writer, iteration, optimizer=None,
                   coefs=None, args=None):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    n_examples = 0
    n_correct = 0
    n_batches = 0

    total_cross_entropy = 0
    total_cluster_cost = 0
    total_separation_cost = 0
    total_orth_cost = 0

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 60

    logger = logging.getLogger("train")
    logger.info("Start train one epoch")
    it = 0

    for image, label in metric_logger.log_every(dataloader, print_freq, header):
        input = image.cuda()
        target = label.cuda()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, (min_distances, proto_acts, shallow_feas, deep_feas) = model(input)
            del input
            # Compute losses
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            model_without_ddp = model.module if hasattr(model, 'module') else model
            # Clst loss
            cluster_cost = model_without_ddp.get_clst_loss(min_distances, label)
            # Seq loss
            separation_cost = model_without_ddp.get_sep_loss(min_distances, label)
            # Ortho loss
            ortho_cost = model_without_ddp.get_ortho_loss()
            # Consis loss
            consis_cost = model_without_ddp.get_SDFA_loss(proto_acts, shallow_feas, deep_feas, target, consis_thresh=args.consis_thresh)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_orth_cost += ortho_cost.item()

        if coefs is not None:
            if epoch >= args.warmup_epochs:
                loss = (coefs['crs_ent'] * cross_entropy
                    + coefs['clst'] * cluster_cost
                    + coefs['sep'] * separation_cost
                    + coefs['orth'] * ortho_cost
                    + coefs['consis'] * consis_cost)
            else:
                loss = (coefs['crs_ent'] * cross_entropy
                    + coefs['clst'] * cluster_cost
                    + coefs['sep'] * separation_cost
                    + coefs['orth'] * ortho_cost)
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

        # Compute gradient and do SGD step
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Nomalize basis vectors
            model_without_ddp.prototype_vectors.data = F.normalize(model_without_ddp.prototype_vectors, p=2, dim=1).data

            tb_writer.add_scalars(
                main_tag="train/loss",
                tag_scalar_dict={
                    "cls": loss.item(),
                },
                global_step=iteration+it
            )
            it += 1
        else:
            tb_writer.add_scalars(
                main_tag="test/loss",
                tag_scalar_dict={
                    "cls": loss.item(),
                },
                global_step=iteration+it
            )
            it += 1

        # Del input
        del target
        del output
        del predicted
        del min_distances
    
    results_loss = {'cross_entropy': total_cross_entropy / n_batches,
                    'orth_loss': total_orth_cost / n_batches,
                    'accu' : n_correct / n_examples * 100
                    }
    return n_correct / n_examples * 100, results_loss


def train(model, epoch, dataloader, optimizer, tb_writer, iteration, coefs=None, args=None):
    assert(optimizer is not None)

    model.train()
    return _train_or_test(model=model, epoch=epoch, dataloader=dataloader, optimizer=optimizer, tb_writer=tb_writer,
                          iteration=iteration, coefs=coefs, args=args)


def test(model, epoch, dataloader, tb_writer, iteration, coefs=None, args=None):
    model.eval()
    return _train_or_test(model=model, epoch=epoch, dataloader=dataloader, optimizer=None, tb_writer=tb_writer,
                          iteration=iteration, coefs=coefs, args=args)


def warm_only(model):
    if hasattr(model, 'module'):
        model = model.module
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.activation_weight.requires_grad = True
    model.prototype_vectors.requires_grad = True


def joint(model):
    if hasattr(model, 'module'):
        model = model.module
    for p in model.features.parameters():
        p.requires_grad = True
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.activation_weight.requires_grad = True
    model.prototype_vectors.requires_grad = True