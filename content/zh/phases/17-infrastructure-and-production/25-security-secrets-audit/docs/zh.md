# 安全 —— 密钥、API Key 轮换、审计日志与防护栏

> 用集中式 vault（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消灭密钥散落问题。永远不要把凭据存放在配置文件、VCS 中的 env 文件或电子表格里。用 IAM 角色代替静态密钥；CI/CD 使用 OIDC。AI 网关（AI-gateway）模式是 2026 年的解法：应用 → 网关 → 模型提供商，由网关在运行时从 vault 拉取凭据。在 vault 中轮换，所有应用几分钟内自动跟上——无需重新部署，也不用在 Slack 里问「谁有新密钥」。轮换策略 ≤ 90 天；在每次提交上用 TruffleHog / GitGuardian / Gitleaks 扫描。零信任：MFA、SSO、RBAC/ABAC、短期 token、设备姿态检查。PII 脱敏使用实体识别，在转发前屏蔽 PHI/PII；一致性 token 化（Mesh 方法）把敏感值映射到稳定的占位符，让 LLM 保留代码与关系语义。网络出口：把 LLM 服务放在专用 VPC/VNet 子网中，只白名单 `api.openai.com`、`api.anthropic.com` 等地址；阻断其他所有出站流量。2026 年的事故推手：Vercel 供应链攻击，攻击者通过被攻陷的 CI/CD 凭据窃取了数千个客户部署中的环境变量。

**Type:** Learn
**Languages:** Python (stdlib, toy PII-scrubber + audit-log writer)
**Prerequisites:** Phase 17 · 19 (AI Gateways), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## 学习目标

- 列举四种密钥管理反模式（VCS 中的配置文件、硬编码环境变量、电子表格、静态密钥），并说出各自的替代方案。
- 解释「AI 网关从 vault 拉取凭据」这一 2026 年生产标准模式。
- 实现一个一致性 token 化的 PII 脱敏器（相同值 → 相同占位符），让语义得以保留。
- 说出 2026 年 Vercel 供应链事件，以及它对 CI/CD 凭据卫生的教训。

## 问题

一名实习生把带 API key 的 `.env` 提交了。他们很快删掉了文件。但密钥已经进入 git 历史——GitGuardian 扫描发现了它，而你的轮换流程是「在 Slack 通知团队，更新 40 个配置文件，重新部署所有服务」。8 小时后，一半服务在线，另一半还在等部署窗口。

另一件事：用户 prompt 里写着「我的 SSN 是 123-45-6789」。prompt 发往 OpenAI。你签了 BAA，但内部政策要求在转发前屏蔽 PII。你没做。

还有一件：你的 EKS 集群里的 LLM pod 可以访问任意互联网主机。有人通过 DNS 查询把数据外泄到攻击者控制的域名。没有任何东西拦住它。

LLM 服务的安全必须同时处理这三条路径：vault 托管凭据、PII 脱敏、网络出口过滤、审计日志。

## 概念

### 集中式 vault + IAM 角色拉取

**Vault**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。唯一的事实来源。

**IAM 角色**：应用/网关通过自己的 IAM 身份认证，而不是静态密钥。Vault 在 token 生命周期内返回密钥。

**AI 网关模式**：网关在请求时从 vault 拉取 `OPENAI_API_KEY`。在 vault 中轮换后，下一个请求就拿到新密钥。无需重新部署。

### 轮换策略 ≤ 90 天

覆盖所有 API key、vault 根 token、CI/CD 凭据。能自动化的就自动化。手动轮换要记录并跟踪。

### 密钥扫描

- **TruffleHog** —— 基于正则 + 熵值扫描提交。
- **GitGuardian** —— 商业产品，准确率高。
- **Gitleaks** —— 开源，在 CI 中运行。

每次提交都扫描。发现新密钥就阻断 PR。

### 零信任姿态

- 所有账户强制 MFA。
- 通过 SAML/OIDC 实现 SSO。
- 用 RBAC（基于角色）或 ABAC（基于属性）做细粒度访问控制。
- 短期 token（以小时计，而非天）。
- 设备姿态——只允许启用了磁盘加密的公司设备。

### PII / PHI 脱敏

在 prompt 离开你的基础设施之前：

1. 实体识别（spaCy NER、Presidio、商业方案）。
2. 屏蔽匹配到的实体：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. 一致性 token 化（Mesh 方法）：相同值映射到相同占位符，让 LLM 保留关系。
4. 可选：对 LLM 响应做反向映射。

静态正则过滤器能抓住基本模式；NER 能抓住更多。两者都要用。

### 输入 + 输出防护栏

输入：阻断已知越狱、禁止话题；按用户做速率限制。

输出：用正则清洗泄露的密钥（API key 模式、拒绝语境中的邮箱模式），用分类器检测政策违规。

### 网络出口白名单

把 LLM 服务放在专用子网：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、vault 端点。
- 其他所有流量：丢弃。
- DNS 走仅允许名单的解析器（防止 DNS 隧道外泄）。

### 审计日志

每次 LLM 调用的不可变日志，包含：
- 时间戳。
- 用户 / 租户。
- Prompt 哈希（出于隐私不记录原始 prompt）。
- 模型 + 版本。
- Token 数量。
- 成本。
- 响应哈希。
- 任何防护栏触发。

按监管要求保留（SOC 2 一年，HIPAA 六年）。

### 2026 年 Vercel 事件

供应链攻击：被攻陷的 CI/CD 凭据窃取了数千个客户部署中的环境变量。教训：CI/CD 凭据等同于生产凭据。存进 vault。最小化权限范围。积极轮换。

### 应该记住的数字

- 轮换策略：≤ 90 天。
- 每次提交扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD 凭据被攻陷 → 数千客户 env 变量泄露。
- 审计日志保留：SOC 2 = 1 年，HIPAA = 6 年。

```figure
i4-vault-rotation
```

## 使用它

`code/main.py` 实现了一个玩具级 PII 脱敏器，带一致性 token 化和仅追加的审计日志。

## 交付它

本课产出 `outputs/skill-llm-security-plan.md`。给定监管范围与现状，规划 vault 迁移、脱敏器、出口策略和审计日志。

## 练习

1. 运行 `code/main.py`。发送两个引用同一 SSN 的 prompt。确认两者得到相同的占位符。
2. 为一个运行在 EKS 上、调用 OpenAI + Anthropic + Weaviate 的 vLLM 部署设计网络出口策略。
3. 你发现 git 历史里有一个两年前的密钥。正确响应是什么——轮换密钥、清理历史，还是两者都做？说明理由。
4. 你的审计日志每天增长 10 GB。设计保留分层（热 30 天，温 12 个月，冷 6 年）。
5. 论证反向 token 化（把真实值替换回 LLM 响应）是否值得这份复杂度，还是保持占位符可见更好。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Vault | 「密钥存储」 | 集中式凭据管理服务 |
| IAM role | 「基于身份的认证」 | 应用承担的角色；返回短期凭据 |
| OIDC for CI/CD | 「云签发的 token」 | CI 中无静态密钥——通过 OIDC 确认身份 |
| TruffleHog / GitGuardian / Gitleaks | 「密钥扫描器」 | 提交时的密钥检测 |
| RBAC / ABAC | 「访问控制」 | 基于角色 vs 基于属性 |
| PII scrubbing | 「数据脱敏」 | 移除或 token 化敏感实体 |
| Consistent tokenization | 「稳定占位符」 | 相同值 → 每次相同 token |
| Mesh approach | 「Mesh tokenization」 | 保留语义的 token 化模式 |
| Egress whitelist | 「出站允许名单」 | 只有获准域名可达 |
| Audit log | 「不可变历史」 | 用于合规的仅追加记录 |

## 延伸阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) —— PII 检测与匿名化。
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
