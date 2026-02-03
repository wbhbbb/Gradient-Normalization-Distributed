import torch
from torch.distributions import Bernoulli

class Compressor():
    def __init__(self, unbiased):
        self.unbiased = unbiased
        self.contractive = not self.unbiased

    def compress(self, X):
        pass

class Identity(Compressor):
    def __init__(self,ratio_or_k):
        super().__init__(False)

    def compress(self, X):
        return X

class TopK(Compressor):
    def __init__(self, ratio_or_k):
        super().__init__(False)
        self.ratio_or_k = ratio_or_k

    def _get_k_value(self, numel):
        if isinstance(self.ratio_or_k, float) and 0 < self.ratio_or_k < 1:
            # 按比例计算K值
            k = max(1, int(numel * self.ratio_or_k))
        elif isinstance(self.ratio_or_k, int) and self.ratio_or_k > 0:
            # 固定K值
            k = min(self.ratio_or_k, numel)
        else:
            raise ValueError("ratio_or_k must be a positive float (0,1) for ratio or positive int for fixed K")
        return k

    def compress(self, X):
        if isinstance(X, dict):
            compressed_X = {}
            for name, param in X.items():
                if torch.is_tensor(param):
                    flat_param = param.view(-1)
                    numel = flat_param.numel()
                    k = self._get_k_value(numel)
                    
                    if k > 0 and numel > 0:
                        _, idx = torch.topk(torch.abs(flat_param), k, dim=0)
                        mask = torch.zeros_like(flat_param)
                        mask.scatter_(0, idx, 1)
                        compressed_flat = flat_param * mask
                        compressed_param = compressed_flat.view(param.shape)
                        compressed_X[name] = compressed_param
                    else:
                        compressed_X[name] = torch.zeros_like(param)
                else:
                    compressed_X[name] = param
            return compressed_X
            
        elif isinstance(X, (list, tuple)):
            compressed_X = []
            for param in X:
                if torch.is_tensor(param):
                    flat_param = param.view(-1)
                    numel = flat_param.numel()
                    k = self._get_k_value(numel)
                    
                    if k > 0 and numel > 0:
                        _, idx = torch.topk(torch.abs(flat_param), k, dim=0)
                        mask = torch.zeros_like(flat_param)
                        mask.scatter_(0, idx, 1)
                        compressed_flat = flat_param * mask
                        compressed_param = compressed_flat.view(param.shape)
                        compressed_X.append(compressed_param)
                    else:
                        compressed_X.append(torch.zeros_like(param))
                else:
                    compressed_X.append(param)
            return compressed_X
        else:
            flat_X = X.view(-1)
            numel = flat_X.numel()
            k = self._get_k_value(numel)
            
            if k > 0 and numel > 0:
                _, idx = torch.topk(torch.abs(flat_X), k, dim=0)
                mask = torch.zeros_like(flat_X)
                mask.scatter_(0, idx, 1)
                compressed_flat = flat_X * mask
                return compressed_flat.view(X.shape)
            else:
                return torch.zeros_like(X)

class RandK(Compressor):
    def __init__(self, ratio_or_k, shared_randomness=True):
        super().__init__(False)
        self.ratio_or_k = ratio_or_k
        self.share = shared_randomness

    def _get_k_value(self, numel):
        if isinstance(self.ratio_or_k, float) and 0 < self.ratio_or_k < 1:
            k = max(1, int(numel * self.ratio_or_k))
        elif isinstance(self.ratio_or_k, int) and self.ratio_or_k > 0:
            k = min(self.ratio_or_k, numel)
        else:
            raise ValueError("ratio_or_k must be a positive float (0,1) for ratio or positive int for fixed K")
        return k

    def compress(self, X):
        if isinstance(X, dict):
            compressed_X = {}
            for name, param in X.items():
                if torch.is_tensor(param):
                    flat_param = param.view(-1)
                    numel = flat_param.numel()
                    k = self._get_k_value(numel)
                    
                    if k > 0 and numel > 0:
                        if self.share:
                            idx = torch.randperm(numel)[:k]
                            mask = torch.zeros(numel, device=flat_param.device)
                            mask[idx] = 1
                            compressed_flat = flat_param * mask
                        else:
                            idx = torch.randperm(numel)[:k]
                            mask = torch.zeros_like(flat_param)
                            mask.scatter_(0, idx, 1)
                            compressed_flat = flat_param * mask
                        compressed_param = compressed_flat.view(param.shape)
                        compressed_X[name] = compressed_param
                    else:
                        compressed_X[name] = torch.zeros_like(param)
                else:
                    compressed_X[name] = param
            return compressed_X
            
        elif isinstance(X, (list, tuple)):
            compressed_X = []
            for param in X:
                if torch.is_tensor(param):
                    flat_param = param.view(-1)
                    numel = flat_param.numel()
                    k = self._get_k_value(numel)
                    
                    if k > 0 and numel > 0:
                        if self.share:
                            idx = torch.randperm(numel)[:k]
                            mask = torch.zeros(numel, device=flat_param.device)
                            mask[idx] = 1
                            compressed_flat = flat_param * mask
                        else:
                            idx = torch.randperm(numel)[:k]
                            mask = torch.zeros_like(flat_param)
                            mask.scatter_(0, idx, 1)
                            compressed_flat = flat_param * mask
                        compressed_param = compressed_flat.view(param.shape)
                        compressed_X.append(compressed_param)
                    else:
                        compressed_X.append(torch.zeros_like(param))
                else:
                    compressed_X.append(param)
            return compressed_X
        else:
            flat_X = X.view(-1)
            numel = flat_X.numel()
            k = self._get_k_value(numel)
            
            if k > 0 and numel > 0:
                if self.share:
                    idx = torch.randperm(numel)[:k]
                    mask = torch.zeros(numel, device=flat_X.device)
                    mask[idx] = 1
                    compressed_flat = flat_X * mask
                else:
                    idx = torch.randperm(numel)[:k]
                    mask = torch.zeros_like(flat_X)
                    mask.scatter_(0, idx, 1)
                    compressed_flat = flat_X * mask
                return compressed_flat.view(X.shape)
            else:
                return torch.zeros_like(X)

class uRandK(Compressor):
    def __init__(self, ratio_or_k, shared_randomness=True):
        super().__init__(True)
        self.ratio_or_k = ratio_or_k
        self.share = shared_randomness

    def _get_k_value(self, numel):
        if isinstance(self.ratio_or_k, float) and 0 < self.ratio_or_k < 1:
            k = max(1, int(numel * self.ratio_or_k))
        elif isinstance(self.ratio_or_k, int) and self.ratio_or_k > 0:
            k = min(self.ratio_or_k, numel)
        else:
            raise ValueError("ratio_or_k must be a positive float (0,1) for ratio or positive int for fixed K")
        return k

    def compress(self, X):
        if isinstance(X, dict):
            compressed_X = {}
            for name, param in X.items():
                if torch.is_tensor(param):
                    flat_param = param.view(-1)
                    numel = flat_param.numel()
                    k = self._get_k_value(numel)
                    
                    if k > 0 and numel > 0:
                        if self.share:
                            idx = torch.multinomial(torch.ones(numel), k, replacement=False)
                            mask = torch.zeros(numel, device=flat_param.device)
                            mask[idx] = 1
                            compressed_flat = flat_param * mask / k * numel
                        else:
                            idx = torch.multinomial(torch.ones(numel), k, replacement=False)
                            mask = torch.zeros_like(flat_param)
                            mask.scatter_(0, idx, 1)
                            compressed_flat = flat_param * mask / k * numel
                        compressed_param = compressed_flat.view(param.shape)
                        compressed_X[name] = compressed_param
                    else:
                        compressed_X[name] = torch.zeros_like(param)
                else:
                    compressed_X[name] = param
            return compressed_X
            
        elif isinstance(X, (list, tuple)):
            compressed_X = []
            for param in X:
                if torch.is_tensor(param):
                    flat_param = param.view(-1)
                    numel = flat_param.numel()
                    k = self._get_k_value(numel)
                    
                    if k > 0 and numel > 0:
                        if self.share:
                            idx = torch.multinomial(torch.ones(numel), k, replacement=False)
                            mask = torch.zeros(numel, device=flat_param.device)
                            mask[idx] = 1
                            compressed_flat = flat_param * mask / k * numel
                        else:
                            idx = torch.multinomial(torch.ones(numel), k, replacement=False)
                            mask = torch.zeros_like(flat_param)
                            mask.scatter_(0, idx, 1)
                            compressed_flat = flat_param * mask / k * numel
                        compressed_param = compressed_flat.view(param.shape)
                        compressed_X.append(compressed_param)
                    else:
                        compressed_X.append(torch.zeros_like(param))
                else:
                    compressed_X.append(param)
            return compressed_X
        else:
            flat_X = X.view(-1)
            numel = flat_X.numel()
            k = self._get_k_value(numel)
            
            if k > 0 and numel > 0:
                if self.share:
                    idx = torch.multinomial(torch.ones(numel), k, replacement=False)
                    mask = torch.zeros(numel, device=flat_X.device)
                    mask[idx] = 1
                    compressed_flat = flat_X * mask / k * numel
                else:
                    idx = torch.multinomial(torch.ones(numel), k, replacement=False)
                    mask = torch.zeros_like(flat_X)
                    mask.scatter_(0, idx, 1)
                    compressed_flat = flat_X * mask / k * numel
                return compressed_flat.view(X.shape)
            else:
                return torch.zeros_like(X)

    def cal_omega(self, M):
        k = self._get_k_value(M)
        return M / k - 1

class FCC(Compressor):
    def __init__(self, compressor, R):
        super().__init__(compressor.unbiased)
        self.compressor = compressor
        self.R = R

    def compress(self, X):
        if isinstance(X, dict):
            V = {}
            for name, param in X.items():
                if torch.is_tensor(param):
                    V[name] = torch.zeros_like(param)
            
            if self.unbiased:
                for r in range(self.R):
                    V_diff = self.compressor.compress(
                        {name: param - v for name, param, v in zip(X.keys(), X.values(), V.values()) 
                         if torch.is_tensor(param) and torch.is_tensor(v)}
                    )
                    for name in V.keys():
                        if name in V_diff:
                            V[name] = V[name] + V_diff[name] / (1 + self.compressor.cal_omega(V[name].numel()))
                avg_omega = sum([self.compressor.cal_omega(param.numel()) for param in X.values() if torch.is_tensor(param)]) / len([p for p in X.values() if torch.is_tensor(p)])
                scale_factor = 1 - (avg_omega / (1 + avg_omega)) ** self.R
                result = {}
                for name, param in V.items():
                    result[name] = param / scale_factor if scale_factor != 0 else torch.zeros_like(param)
                return result
            else:
                for r in range(self.R):
                    V_diff = self.compressor.compress(
                        {name: param - V[name] for name, param in X.items() if torch.is_tensor(param)}
                    )
                    for name in V.keys():
                        if name in V_diff:
                            V[name] = V[name] + V_diff[name]
                return V
                
        elif isinstance(X, (list, tuple)):
            V = [torch.zeros_like(param) if torch.is_tensor(param) else param for param in X]
            
            if self.unbiased:
                for r in range(self.R):
                    X_V_diff = [param - v for param, v in zip(X, V) if torch.is_tensor(param)]
                    V_diff = self.compressor.compress(X_V_diff)
                    for i, (v, v_diff) in enumerate(zip(V, V_diff)):
                        if torch.is_tensor(v):
                            omega = self.compressor.cal_omega(v.numel())
                            V[i] = v + v_diff / (1 + omega)
                avg_omega = sum([self.compressor.cal_omega(param.numel()) for param in X if torch.is_tensor(param)]) / len([p for p in X if torch.is_tensor(p)])
                scale_factor = 1 - (avg_omega / (1 + avg_omega)) ** self.R
                return [v / scale_factor if scale_factor != 0 else torch.zeros_like(param) 
                       for v, param in zip(V, X) if torch.is_tensor(param)]
            else:
                for r in range(self.R):
                    X_V_diff = [param - v for param, v in zip(X, V) if torch.is_tensor(param)]
                    V_diff = self.compressor.compress(X_V_diff)
                    for i, (v, v_diff) in enumerate(zip(V, V_diff)):
                        if torch.is_tensor(v):
                            V[i] = v + v_diff
                return V
        else:
            V = torch.zeros_like(X)
            
            if self.unbiased:
                omega = self.compressor.cal_omega(X.numel())
                for r in range(self.R):
                    V = V + self.compressor.compress(X - V) / (1 + omega)
                scale_factor = 1 - (omega / (1 + omega)) ** self.R
                return V / scale_factor if scale_factor != 0 else torch.zeros_like(X)
            else:
                for r in range(self.R):
                    V = V + self.compressor.compress(X - V)
                return V

    def cal_omega(self, M):
        assert self.unbiased
        k = self.compressor._get_k_value(M) if hasattr(self.compressor, '_get_k_value') else min(self.compressor.K, M)
        omega = M / k - 1
        return omega * (omega / (1 + omega)) ** self.R

class NaturalCompression(Compressor):
    def __init__(self):
        super().__init__(True)

    def compress(self, X):
        if isinstance(X, dict):
            compressed_X = {}
            for name, param in X.items():
                if torch.is_tensor(param):
                    mantissa, exponent = torch.frexp(param)
                    sign = torch.sign(param)
                    unsigned_mantissa = mantissa * sign
                    p = (mantissa * sign * 2 - 1).clamp(min=0)
                    shift = Bernoulli(p).sample()
                    compressed_param = sign * 2.0 ** (exponent + shift - 1)
                    compressed_X[name] = compressed_param
                else:
                    compressed_X[name] = param
            return compressed_X
            
        elif isinstance(X, (list, tuple)):
            compressed_X = []
            for param in X:
                if torch.is_tensor(param):
                    mantissa, exponent = torch.frexp(param)
                    sign = torch.sign(param)
                    unsigned_mantissa = mantissa * sign
                    p = (mantissa * sign * 2 - 1).clamp(min=0)
                    shift = Bernoulli(p).sample()
                    compressed_param = sign * 2.0 ** (exponent + shift - 1)
                    compressed_X.append(compressed_param)
                else:
                    compressed_X.append(param)
            return compressed_X
        else:
            mantissa, exponent = torch.frexp(X)
            sign = torch.sign(X)
            unsigned_mantissa = mantissa * sign
            p = (mantissa * sign * 2 - 1).clamp(min=0)
            shift = Bernoulli(p).sample()
            return sign * 2.0 ** (exponent + shift - 1)

class RandomQuantization(Compressor):
    def __init__(self, s):
        super().__init__(True)
        self.s = s
        self.llist = torch.tensor([1] + [3] * 2 + [6] * 4 + [7] * 8 + [11] * 16 + [12] * 32)
        self.name = 'RandomQuantization'

    def compress(self, X):
        if isinstance(X, dict):
            compressed_X = {}
            total_bits = 0
            for name, param in X.items():
                if torch.is_tensor(param):
                    norm_0 = torch.norm(param, dim=list(range(1, param.dim())), keepdim=True)
                    norm = norm_0 + (norm_0 == 0).float()
                    signs = torch.sign(param)
                    X_abs = torch.abs(param)
                    X_normalized = X_abs / norm
                    X_scaled = X_normalized * self.s
                    
                    lb = torch.floor(X_scaled)
                    p = X_scaled - lb
                    random_matrix = Bernoulli(p).sample()
                    quantized_matrix = lb + random_matrix
                    
                    compressed_param = quantized_matrix * norm_0 * signs / self.s
                    compressed_X[name] = compressed_param
                    
                    levels = quantized_matrix.long()
                    bits = self.llist[torch.clamp(levels, 0, len(self.llist)-1)].sum().item()
                    total_bits += bits
                else:
                    compressed_X[name] = param
            
            avg_bits = total_bits / len([p for p in X.values() if torch.is_tensor(p)])
            return compressed_X, avg_bits + 64  
            
        elif isinstance(X, (list, tuple)):
            compressed_X = []
            total_bits = 0
            for param in X:
                if torch.is_tensor(param):
                    norm_0 = torch.norm(param, dim=list(range(1, param.dim())), keepdim=True)
                    norm = norm_0 + (norm_0 == 0).float()
                    signs = torch.sign(param)
                    X_abs = torch.abs(param)
                    X_normalized = X_abs / norm
                    X_scaled = X_normalized * self.s
                    
                    lb = torch.floor(X_scaled)
                    p = X_scaled - lb
                    random_matrix = Bernoulli(p).sample()
                    quantized_matrix = lb + random_matrix
                    
                    compressed_param = quantized_matrix * norm_0 * signs / self.s
                    compressed_X.append(compressed_param)
                    
                    levels = quantized_matrix.long()
                    bits = self.llist[torch.clamp(levels, 0, len(self.llist)-1)].sum().item()
                    total_bits += bits
                else:
                    compressed_X.append(param)
            
            avg_bits = total_bits / len([p for p in X if torch.is_tensor(p)])
            return compressed_X, avg_bits + 64
        else:
            norm_0 = torch.norm(X, dim=list(range(1, X.dim())), keepdim=True)
            norm = norm_0 + (norm_0 == 0).float()
            signs = torch.sign(X)
            X_abs = torch.abs(X)
            X_normalized = X_abs / norm
            X_scaled = X_normalized * self.s
            
            lb = torch.floor(X_scaled)
            p = X_scaled - lb
            random_matrix = Bernoulli(p).sample()
            quantized_matrix = lb + random_matrix
            
            compressed_X = quantized_matrix * norm_0 * signs / self.s
            
            levels = quantized_matrix.long()
            bits = self.llist[torch.clamp(levels, 0, len(self.llist)-1)].sum(dim=list(range(1, X.dim()))).mean().item()
            return compressed_X, bits + 64