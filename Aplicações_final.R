# PACOTES 

library(survival)
library(bbmle)
library(broom)
library(VGAM)

erf <- function(x) 2 * pnorm(x * sqrt(2)) - 1

# FUNÇÃO MODELOS P

modelos <- function(tempos_raw, cens, modelo = 'MOMB', k = 2) {
  t_max <- max(tempos_raw)
  tempos <- tempos_raw / t_max
  n <- length(tempos_raw)
  
  if (modelo == 'Standard Mixture Exponential') {
    densidade <- function(x, p, q) p * q * exp(-q * x)
    acumulada <- function(x, p, q) p - p * exp(-q * x)
    st <- list(p = 0.5, q = 1.0)
    lower_b <- c(p = 1e-5, q = 1e-5)
    upper_b <- c(p = 0.999, q = Inf)
    loglike <- function(p, q) {
      f <- densidade(tempos, p, q)
      s <- 1 - acumulada(tempos, p, q)
      if (any(is.na(f)) || any(f <= 0) || any(is.na(s)) || any(s <= 0)) return(1e10)
      -sum(cens * log(f) + (1 - cens) * log(s))
    }
  }
  
  if (modelo == 'MO-Exp') {
    densidade <- function(x, r, a) {
      f <- dexp(x, a); G <- pexp(x, a)
      r * f / ((1 - (1 - r) * (1 - G))^2)
    }
    acumulada <- function(x, r, a) {
      G <- pexp(x, a)
      1 - r * (1 - G) / (1 - (1 - r) * (1 - G))
    }
    st <- list(r = 0.5, a = 1.0)
    lower_b <- c(r = 1e-5, a = 1e-5)
    upper_b <- c(r = Inf, a = Inf)
    loglike <- function(r, a) {
      f <- densidade(tempos, r, a)
      s <- 1 - acumulada(tempos, r, a)
      if (any(is.na(f)) || any(f <= 0) || any(is.na(s)) || any(s <= 0)) return(1e10)
      -sum(cens * log(f) + (1 - cens) * log(s))
    }
  }
  
  if (modelo == 'Gompertz') {
    densidade <- function(x, a, b) b * exp(a * x) * exp(-(exp(a * x) - 1) * b / a)
    acumulada <- function(x, a, b) 1 - exp(-(exp(a * x) - 1) * b / a)
    st <- list(a = -1.0, b = 1.0)
    lower_b <- c(a = -Inf, b = 1e-5)
    upper_b <- c(a = Inf, b = Inf)
    loglike <- function(a, b) {
      f <- densidade(tempos, a, b)
      s <- 1 - acumulada(tempos, a, b)
      if (any(is.na(f)) || any(f <= 0) || any(is.na(s)) || any(s <= 0)) return(1e10)
      -sum(cens * log(f) + (1 - cens) * log(s))
    }
  }
  
  if (modelo == 'Inverse Gaussian') {
    densidade <- function(x, a, b) exp(-b * (x - a)^2 / (2 * a^2 * x)) * (b / (2 * pi * x^3))^(1/2)
    acumulada <- function(x, a, b) pnorm(sqrt(b / x) * (x / a - 1)) + exp(2 * a / b) * pnorm(-sqrt(b / x) * (x / a + 1))
    st <- list(a = 0.5, b = 0.5)
    lower_b <- c(a = 1e-5, b = 1e-5)
    upper_b <- c(a = Inf, b = Inf)
    loglike <- function(a, b) {
      f <- densidade(tempos, a, b)
      s <- 1 - acumulada(tempos, a, b)
      if (any(is.na(f)) || any(f <= 0) || any(is.na(s)) || any(s <= 0)) return(1e10)
      -sum(cens * log(f) + (1 - cens) * log(s))
    }
  }
  
  if (modelo == 'MOMB') {
    S_aux <- function(x, a) {
      z <- x / (sqrt(2) * a)
      1 - erf(z) + sqrt(2/pi) * (x / a) * exp(-x^2 / (2 * a^2))
    }
    f_aux <- function(x, a) {
      sqrt(2/pi) * (x^2 / a^3) * exp(-x^2 / (2 * a^2))
    }
    densidade <- function(x, a, b) {
      s0 <- S_aux(x, a)
      f0 <- f_aux(x, a)
      b * f0 / ((1 - (1 - b) * s0)^2)
    }
    acumulada <- function(x, a, b) {
      s0 <- S_aux(x, a)
      1 - (b * s0 / (1 - (1 - b) * s0))
    }
    st <- list(a = -0.25, b = -0.5)
    lower_b <- c(a = -Inf, b = -Inf)
    upper_b <- c(a = -1e-4, b = -1e-4)
    loglike <- function(a, b) {
      f <- densidade(tempos, a, b)
      s <- 1 - acumulada(tempos, a, b)
      if (any(is.na(f)) || any(f <= 0) || any(is.na(s)) || any(s <= 0)) return(1e10)
      -sum(cens * log(f) + (1 - cens) * log(s))
    }
  }
  
  emv <- mle2(loglike, start = st, method = "L-BFGS-B", lower = lower_b, upper = upper_b)
  
  loglik_val <- as.numeric(logLik(emv))
  aic  <- -2 * loglik_val + 2 * k
  bic  <- -2 * loglik_val + k * log(n)
  aicc <- aic + (2 * k * (k + 1)) / (n - k - 1)
  caic <- -2 * loglik_val + k * (log(n) + 1)
  
  medidas <- c(AIC = aic, BIC = bic, CAIC = caic, AICc = aicc)
  return(list(medidas = medidas, EMV = emv, t_max = t_max, n = n))
}

extrair_tabela_modelo <- function(fit, nome_modelo) {
  coef_df <- as.data.frame(broom::tidy(fit$EMV))
  med <- fit$medidas
  
  coef_df$ci_inf <- coef_df$estimate - 1.96 * coef_df$std.error
  coef_df$ci_sup <- coef_df$estimate + 1.96 * coef_df$std.error
  
  # Converte para numérico simples (remove os nomes das linhas)
  aic_val  <- as.numeric(med["AIC"])
  bic_val  <- as.numeric(med["BIC"])
  aicc_val <- as.numeric(med["AICc"])
  caic_val <- as.numeric(med["CAIC"])
  
  df_res <- data.frame(
    Modelo    = c(nome_modelo, rep("", nrow(coef_df) - 1)),
    Parametro = coef_df$term,
    Est       = format(round(coef_df$estimate, 2), nsmall = 2),
    EP        = format(round(coef_df$std.error, 2), nsmall = 2),
    IC_inf    = format(round(coef_df$ci_inf, 2), nsmall = 2),
    IC_sup    = format(round(coef_df$ci_sup, 2), nsmall = 2),
    AIC       = c(format(round(aic_val, 2), nsmall = 2), rep("", nrow(coef_df) - 1)),
    BIC       = c(format(round(bic_val, 2), nsmall = 2), rep("", nrow(coef_df) - 1)),
    AICc      = c(format(round(aicc_val, 2), nsmall = 2), rep("", nrow(coef_df) - 1)),
    CAIC      = c(format(round(caic_val, 2), nsmall = 2), rep("", nrow(coef_df) - 1)),
    stringsAsFactors = FALSE
  )
  
  # Linha da fração de cura (p) no MOMB
  if (nome_modelo == "MOMB") {
    b_hat <- coef_df$estimate[coef_df$term == "b"]
    se_b  <- coef_df$std.error[coef_df$term == "b"]
    
    p_hat <- 2 * b_hat / (2 * b_hat - 1)
    se_p  <- abs(-2 / (2 * b_hat - 1)^2) * se_b
    ic_inf_p <- p_hat - 1.96 * se_p
    ic_sup_p <- p_hat + 1.96 * se_p
    
    linha_p <- data.frame(
      Modelo    = "",
      Parametro = "p (cura)",
      Est       = format(round(p_hat, 2), nsmall = 2),
      EP        = format(round(se_p, 2), nsmall = 2),
      IC_inf    = format(round(ic_inf_p, 2), nsmall = 2),
      IC_sup    = format(round(ic_sup_p, 2), nsmall = 2),
      AIC       = "", BIC = "", AICc = "", CAIC = "",
      stringsAsFactors = FALSE
    )
    df_res <- rbind(df_res, linha_p)
  }
  
  rownames(df_res) <- NULL
  return(df_res)
}

# APLICAÇÃO 1 (Hodgkin, n=15)

t1_app1 <- c(1.05, 2.92, 3.61, 4.2, 4.49, 6.72, 7.31, 9.08, 9.11, 14.49, 16.85, 18.82, 26.59, 30.26, 41.34)
c1_app1 <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0)

fit_momb_app1 <- modelos(t1_app1, c1_app1, modelo = 'MOMB')
fit_gomp_app1 <- modelos(t1_app1, c1_app1, modelo = 'Gompertz')
fit_moex_app1 <- modelos(t1_app1, c1_app1, modelo = 'MO-Exp')
fit_invg_app1 <- modelos(t1_app1, c1_app1, modelo = 'Inverse Gaussian')
fit_smex_app1 <- modelos(t1_app1, c1_app1, modelo = 'Standard Mixture Exponential')

tabela_final_app1 <- rbind(
  extrair_tabela_modelo(fit_momb_app1, "MOMB"),
  extrair_tabela_modelo(fit_gomp_app1, "Gompertz"),
  extrair_tabela_modelo(fit_moex_app1, "MO-Exp"),
  extrair_tabela_modelo(fit_invg_app1, "Inverse Gaussian"),
  extrair_tabela_modelo(fit_smex_app1, "Standard Mixture Exponential")
)
tabela_final_app1

      
      
# EXECUÇÃO APLICAÇÃO 2 (Breast Cancer, n=36)

t1_app2 <- c(19,25,30,34,37,46,47,51,56,57,61,66,67,74,78,86,122,123,130,130,133,134,136,141,143,148,151,152,153,154,156,162,164,165,182,189)
c1_app2 <- c(rep(1,16), rep(0,20))
fit_momb_app2 <- modelos(t1_app2, c1_app2, modelo = 'MOMB')
fit_gomp_app2 <- modelos(t1_app2, c1_app2, modelo = 'Gompertz')
fit_moex_app2 <- modelos(t1_app2, c1_app2, modelo = 'MO-Exp')
fit_invg_app2 <- modelos(t1_app2, c1_app2, modelo = 'Inverse Gaussian')
fit_smex_app2 <- modelos(t1_app2, c1_app2, modelo = 'Standard Mixture Exponential')

tabela_final_app2 <- rbind(
  extrair_tabela_modelo(fit_momb_app2, "MOMB"),
  extrair_tabela_modelo(fit_gomp_app2, "Gompertz"),
  extrair_tabela_modelo(fit_moex_app2, "MO-Exp"),
  extrair_tabela_modelo(fit_invg_app2, "Inverse Gaussian"),
  extrair_tabela_modelo(fit_smex_app2, "Standard Mixture Exponential")
)
tabela_final_app2





# GRÁFICOs DE SOBREVIVÊNCIA
gerar_grafico_2x2 <- function(t1, c1, fit_momb, fit_gomp, fit_invg, fit_moex, fit_smex, titulo_arquivo = "grafico_2x2.pdf") {
  t_norm <- t1 / max(t1)
  ekm <- survfit(Surv(t_norm, c1) ~ 1)
  t_grid <- seq(0.001, 1.0, length.out = 500)
  
  # Curva MOMB
  a_momb <- coef(fit_momb$EMV)["a"]; b_momb <- coef(fit_momb$EMV)["b"]
  S_aux <- function(x, a) {
    z <- x / (sqrt(2) * a)
    1 - erf(z) + sqrt(2/pi) * (x / a) * exp(-x^2 / (2 * a^2))
  }
  s0 <- S_aux(t_grid, a_momb)
  S_momb <- b_momb * s0 / (1 - (1 - b_momb) * s0)
  aic_momb <- sprintf("%.2f", fit_momb$medidas["AIC"])
  
  # Curva Gompertz 
  a_gomp <- coef(fit_gomp$EMV)["a"]; b_gomp <- coef(fit_gomp$EMV)["b"]
  S_gomp <- exp(-(exp(a_gomp * t_grid) - 1) * b_gomp / a_gomp)
  aic_gomp <- sprintf("%.2f", fit_gomp$medidas["AIC"])
  
  # Curva Inverse Gaussian 
  a_invg <- coef(fit_invg$EMV)["a"]; b_invg <- coef(fit_invg$EMV)["b"]
  F_invg <- pnorm(sqrt(b_invg / t_grid) * (t_grid / a_invg - 1)) + 
    exp(2 * a_invg / b_invg) * pnorm(-sqrt(b_invg / t_grid) * (t_grid / a_invg + 1))
  F_invg[t_grid == 0] <- 0
  S_invg <- 1 - F_invg
  aic_invg <- sprintf("%.2f", fit_invg$medidas["AIC"])
  
  # Curva MO-Exponential 
  r_moex <- coef(fit_moex$EMV)["r"]; a_moex <- coef(fit_moex$EMV)["a"]
  S0_moex <- exp(-a_moex * t_grid)
  S_moex <- r_moex * S0_moex / (1 - (1 - r_moex) * S0_moex)
  aic_moex <- sprintf("%.2f", fit_moex$medidas["AIC"])
  
  # Curva Standard Mixture Exponential
  p_smex <- coef(fit_smex$EMV)["p"]; q_smex <- coef(fit_smex$EMV)["q"]
  S_smex <- (1 - p_smex) + p_smex * exp(-q_smex * t_grid)
  aic_smex <- sprintf("%.2f", fit_smex$medidas["AIC"])
 
  pdf(titulo_arquivo, width = 10, height = 7)
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  
   plot(ekm, mark.time = FALSE, col = "black", lwd = 1.8, xlab = "", ylab = "", ylim = c(0, 1), xlim = c(0, 1))
  lines(t_grid, S_momb, col = "#DB4437", lwd = 2)     # Vermelho
  lines(t_grid, S_gomp, col = "#4285F4", lwd = 1.8)   # Azul
  legend("topright", 
         legend = c(paste("MOMB AIC:", aic_momb), 
                    paste("Gompertz AIC:", aic_gomp), 
                    "Kaplan Meier"),
         text.col = c("#DB4437", "#4285F4", "black"), 
         col = c(NA, NA, NA), bty = "o", box.col = "black", cex = 0.8)
  
  plot(ekm, mark.time = FALSE, col = "black", lwd = 1.8, xlab = "", ylab = "", ylim = c(0, 1), xlim = c(0, 1))
  lines(t_grid, S_momb, col = "#DB4437", lwd = 2)
  lines(t_grid, S_invg, col = "#4285F4", lwd = 1.8)
  legend("topright", 
         legend = c(paste("MOMB AIC:", aic_momb), 
                    paste("Inverse Gaussian AIC:", aic_invg), 
                    "Kaplan Meier"),
         text.col = c("#DB4437", "#4285F4", "black"), 
         col = c(NA, NA, NA), bty = "o", box.col = "black", cex = 0.8)
  
  plot(ekm, mark.time = FALSE, col = "black", lwd = 1.8, xlab = "", ylab = "", ylim = c(0, 1), xlim = c(0, 1))
  lines(t_grid, S_momb, col = "#DB4437", lwd = 2)
  lines(t_grid, S_moex, col = "#4285F4", lwd = 1.8)
  legend("topright", 
         legend = c(paste("MOMB AIC:", aic_momb), 
                    paste("Marshall Olkin Exponential AIC:", aic_moex), 
                    "Kaplan Meier"),
         text.col = c("#DB4437", "#4285F4", "black"), 
         col = c(NA, NA, NA), bty = "o", box.col = "black", cex = 0.8)
  
  plot(ekm, mark.time = FALSE, col = "black", lwd = 1.8, xlab = "", ylab = "", ylim = c(0, 1), xlim = c(0, 1))
  lines(t_grid, S_momb, col = "#DB4437", lwd = 2)
  lines(t_grid, S_smex, col = "#4285F4", lwd = 1.8)
  legend("topright", 
         legend = c(paste("MOMB AIC:", aic_momb), 
                    paste("Standard Mixture Exponential AIC:", aic_smex), 
                    "Kaplan Meier"),
         text.col = c("#DB4437", "#4285F4", "black"), 
         col = c(NA, NA, NA), bty = "o", box.col = "black", cex = 0.8)
  dev.off()
  cat("Gráfico salvo com sucesso em:", titulo_arquivo, "\n")
}

gerar_grafico_2x2(t1_app1, c1_app1, fit_momb_app1, fit_gomp_app1, fit_invg_app1, fit_moex_app1, fit_smex_app1, "grafico_2x2_aplicacao1.pdf")
gerar_grafico_2x2(t1_app2, c1_app2, fit_momb_app2, fit_gomp_app2, fit_invg_app2, fit_moex_app2, fit_smex_app2, "grafico_2x2_aplicacao2.pdf")


# GRÁFICO 2x2 NA ESCALA ORIGINAL DO TEMPO
gerar_grafico_2x2_tempo_original <- function(t1, c1, fit_momb, fit_gomp, fit_invg, fit_moex, fit_smex, titulo_arquivo = "grafico_2x2_tempo_original.pdf", unidade_tempo = "Time (months)") {
  
  t_max <- max(t1)
  ekm <- survfit(Surv(t1, c1) ~ 1)
  t_grid <- seq(0.001, t_max, length.out = 500)
  t_star <- t_grid / t_max
  
  #  Curva MOMB
  a_momb <- coef(fit_momb$EMV)["a"]; b_momb <- coef(fit_momb$EMV)["b"]
  S_aux <- function(x, a) {
    z <- x / (sqrt(2) * a)
    1 - erf(z) + sqrt(2/pi) * (x / a) * exp(-x^2 / (2 * a^2))
  }
  s0 <- S_aux(t_star, a_momb)
  S_momb <- b_momb * s0 / (1 - (1 - b_momb) * s0)
  aic_momb <- sprintf("%.2f", fit_momb$medidas["AIC"])
  
  # Curva Gompertz
  a_gomp <- coef(fit_gomp$EMV)["a"]; b_gomp <- coef(fit_gomp$EMV)["b"]
  S_gomp <- exp(-(exp(a_gomp * t_star) - 1) * b_gomp / a_gomp)
  aic_gomp <- sprintf("%.2f", fit_gomp$medidas["AIC"])
  
  # Curva Inverse Gaussian
  a_invg <- coef(fit_invg$EMV)["a"]; b_invg <- coef(fit_invg$EMV)["b"]
  F_invg <- pnorm(sqrt(b_invg / t_star) * (t_star / a_invg - 1)) + 
    exp(2 * a_invg / b_invg) * pnorm(-sqrt(b_invg / t_star) * (t_star / a_invg + 1))
  F_invg[t_star == 0] <- 0
  S_invg <- 1 - F_invg
  aic_invg <- sprintf("%.2f", fit_invg$medidas["AIC"])
  
  # Curva MO-Exponential
  r_moex <- coef(fit_moex$EMV)["r"]; a_moex <- coef(fit_moex$EMV)["a"]
  S0_moex <- exp(-a_moex * t_star)
  S_moex <- r_moex * S0_moex / (1 - (1 - r_moex) * S0_moex)
  aic_moex <- sprintf("%.2f", fit_moex$medidas["AIC"])
  
  # Curva Standard Mixture Exponential
  p_smex <- coef(fit_smex$EMV)["p"]; q_smex <- coef(fit_smex$EMV)["q"]
  S_smex <- (1 - p_smex) + p_smex * exp(-q_smex * t_star)
  aic_smex <- sprintf("%.2f", fit_smex$medidas["AIC"])
  # Configurar arquivo PDF / janela gráfica
  pdf(titulo_arquivo, width = 10, height = 7)
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  
  plot(ekm, mark.time = FALSE, col = "black", lwd = 1.8, xlab = unidade_tempo, ylab = "Survival", 
       ylim = c(0, 1), xlim = c(0, t_max))
  lines(t_grid, S_momb, col = "#DB4437", lwd = 2)     # Vermelho
  lines(t_grid, S_gomp, col = "#4285F4", lwd = 1.8)   # Azul
  legend("topright", 
         legend = c(paste("MOMB AIC:", aic_momb), 
                    paste("Gompertz AIC:", aic_gomp), 
                    "Kaplan Meier"),
         text.col = c("#DB4437", "#4285F4", "black"), 
         col = c(NA, NA, NA), bty = "o", box.col = "black", cex = 0.8)
 
  plot(ekm, mark.time = FALSE, col = "black", lwd = 1.8, xlab = unidade_tempo, ylab = "Survival", 
       ylim = c(0, 1), xlim = c(0, t_max))
  lines(t_grid, S_momb, col = "#DB4437", lwd = 2)
  lines(t_grid, S_invg, col = "#4285F4", lwd = 1.8)
  legend("topright", 
         legend = c(paste("MOMB AIC:", aic_momb), 
                    paste("Inverse Gaussian AIC:", aic_invg), 
                    "Kaplan Meier"),
         text.col = c("#DB4437", "#4285F4", "black"), 
         col = c(NA, NA, NA), bty = "o", box.col = "black", cex = 0.8)

  plot(ekm, mark.time = FALSE, col = "black", lwd = 1.8, xlab = unidade_tempo, ylab = "Survival", 
       ylim = c(0, 1), xlim = c(0, t_max))
  lines(t_grid, S_momb, col = "#DB4437", lwd = 2)
  lines(t_grid, S_moex, col = "#4285F4", lwd = 1.8)
  legend("topright", 
         legend = c(paste("MOMB AIC:", aic_momb), 
                    paste("Marshall Olkin Exponential AIC:", aic_moex), 
                    "Kaplan Meier"),
         text.col = c("#DB4437", "#4285F4", "black"), 
         col = c(NA, NA, NA), bty = "o", box.col = "black", cex = 0.8)

  plot(ekm, mark.time = FALSE, col = "black", lwd = 1.8, xlab = unidade_tempo, ylab = "Survival", 
       ylim = c(0, 1), xlim = c(0, t_max))
  lines(t_grid, S_momb, col = "#DB4437", lwd = 2)
  lines(t_grid, S_smex, col = "#4285F4", lwd = 1.8)
  legend("topright", 
         legend = c(paste("MOMB AIC:", aic_momb), 
                    paste("Standard Mixture Exponential AIC:", aic_smex), 
                    "Kaplan Meier"),
         text.col = c("#DB4437", "#4285F4", "black"), 
         col = c(NA, NA, NA), bty = "o", box.col = "black", cex = 0.8)
  dev.off()
  cat("Gráfico em tempo original salvo em:", titulo_arquivo, "\n")
}

gerar_grafico_2x2_tempo_original(t1_app1, c1_app1, fit_momb_app1, fit_gomp_app1, fit_invg_app1, fit_moex_app1, fit_smex_app1, "grafico_2x2_tempo_original_app1.pdf", "Time (months)")

gerar_grafico_2x2_tempo_original(t1_app2, c1_app2, fit_momb_app2, fit_gomp_app2, fit_invg_app2, fit_moex_app2, fit_smex_app2, "grafico_2x2_tempo_original_app2.pdf", "Time (months)")


library(TH.data)
data(GBSG2)

t1_gbsg <- GBSG2$time
c1_gbsg <- GBSG2$cens

fit_momb_app5 <- modelos(t1_gbsg,c1_gbsg,"MOMB")
fit_gomp_app5 <- modelos(t1_gbsg,c1_gbsg,"Gompertz")
fit_invg_app5 <- modelos(t1_gbsg,c1_gbsg,"Inverse Gaussian")
fit_moex_app5 <- modelos(t1_gbsg,c1_gbsg,"MO-Exp")
fit_smex_app5 <- modelos(t1_gbsg,c1_gbsg,"Standard Mixture Exponential")


tabela_final_app5 <- rbind(
  extrair_tabela_modelo(fit_momb_app5, "MOMB"),
  extrair_tabela_modelo(fit_gomp_app5, "Gompertz"),
  extrair_tabela_modelo(fit_moex_app5, "MO-Exp"),
  extrair_tabela_modelo(fit_invg_app5, "Inverse Gaussian"),
  extrair_tabela_modelo(fit_smex_app5, "Standard Mixture Exponential")
)
tabela_final_app5

gerar_grafico_2x2_tempo_original(t1_gbsg, c1_gbsg, fit_momb_app5, fit_gomp_app5, fit_invg_app5, fit_moex_app5, fit_smex_app5, "grafico_2x2_tempo_original_app5.pdf", "Time (days)")






####################### Aplica??o 1 ###################
#pacotes utilizados
library(survival)
library(bbmle)
library(msm)
library(VGAM) #funcao 'erf'

#carregando dados
# OK Bartolucci and Dickey 1977, grupoB

t1=c(1.05,2.92,3.61,4.2,4.49,6.72,7.31,9.08,9.11,14.49,16.85,18.82,26.59,30.26,41.34)
c1=c(1   ,   1,   1,  1,   1,   1,   1,   1,   1,    0,    1,    0,    0,    0,    0)


set.seed(2019)
# Distribution's parameters estimation

modelos = function (varnum, modelo = 'MOMB',k=k, cov_a = FALSE, cov_b = FALSE){
  tempos = t1/max(t1)
  cens = c1
  
  
  if(modelo == 'Standard Mixture Exponential'){
    densidade = function(x, p, q){
      p*q*exp(-q*x)}
    acumulada = function(x, p, q) {
      p- p*exp(-q*x)
    }
  }
  
  
  if(modelo == 'Promotion Time Exponential'){
    densidade = function(x, p, a){
      f = dexp(x, a)
      G = pexp(x, a)
      return( p*f*exp(-p*G) )}
    acumulada = function(x, p, a) {
      G = pexp(x, a)
      return(1 - exp(-p*G))
    }
  } 
  
  if(modelo == 'MO-Exp'){
    densidade = function(x, r, a){
      f = dexp(x, a)
      G = pexp(x, a)
      return(r*f/((1 - (1 - r)*(1 - G))^2))
    }
    acumulada = function(x, r, a) {
      G = pexp(x, a)
      return(1 - r*(1 - G)/(1 - (1 - r)*(1 - G)))
    } 
  }
  
  if(modelo == 'Gompertz'){
    densidade = function(x, a, b) b*exp(a*x)*exp(-(exp(a*x) - 1)*b/a)
    #densidade = function(x, a, b) a*b*exp(b)*exp(a*x)*exp(-b*exp(a*x))
    acumulada = function(x, a, b) 1 - exp(-(exp(a*x) - 1)*b/a) 
    #acumulada = function(x, a, b) 1 - exp(-b*(exp(a*x)-1))
  }
  
  if(modelo == 'Inverse Gaussian'){
    #densidade = function(x, a, b) exp(-(1 - a*x)^2/(2*b*x))/(sqrt(2*b*pi*x^3))  
    #acumulada = function(x, a, b) pnorm((-1+a*x)/(sqrt(b*x)))+exp(2*a/b)*pnorm((-1-a*x)/(sqrt(b*x)))
    densidade = function(x, a, b) exp(-b*(x-a)^2/(2*a^2*x))*(b/(2*pi*x^3))^(1/2)  
    acumulada = function(x, a, b) pnorm(sqrt(b/x)*(x/a - 1)) + exp(2*a/b)*pnorm(-sqrt(b/x)*(x/a + 1))  
  }
  
  if(modelo == 'MOMB'){
    densidade_base = function(x, a) a^(-3)*x^2*exp(-x^2/(2*a^2))*sqrt(2/pi)
    acumulada_base = function(x, a) erf(x/(sqrt(2)*a))-(x*exp(-(x^2)/(2*a^2))*sqrt(2/(pi)))/a
    
    densidade = function(x, a, b) b*densidade_base(x = x, a = a)/((1-(1-b)*(1-acumulada_base(x = x, a = a)))^2)
    #acumulada = function(x, a, b) 1-((b*(1-acumulada_base(x = x, a = a)))/((1-(1-b)*(1-acumulada_base(x = x, a = a))))) 
    acumulada = function(x, a, b) ((acumulada_base(x = x, a = a))/((1-(1-b)*(1-acumulada_base(x = x, a = a))))) 
  }
  
  # Fun??o de logverossimilhan?a sem as covari?veis nos par?metros
  loglike = function(a ,b){         # tempos e 'cens' = censuras
    f = densidade(tempos, a ,b)
    s = 1 - acumulada(tempos, a ,b)
    logl = sum((cens*log(f)) + ((1 - cens)*log(s)))
    return(-logl)
  } 
  
  #Estimador de Maxima Verossimilhan?a
  library(bbmle)
  st = list(a = runif(1), b = runif(1))
  emv = mle2(loglike, start = st, method = 'BFGS')
  a=AIC(emv)
  aic= -2*(logLik(emv)) +2*(2)
  AICc=-2*(logLik(emv)) +2*(2)+ 2*(2*(2+1))/(length( t1)-2-1)
  bic= -2*(logLik(emv))+2*log(length( t1))
  CAIC= 0.5 * aic + (1 - 0.5) * bic 
  return(list(AIC = a,medidas = c(aic,bic,CAIC,AICc), EMV = emv))
}


tempos = t1/max(t1)
cens = c1

MOMB = modelos(modelo = 'MOMB', k=k)
MOMBcoef<-tidy(MOMB$EMV)
MOMBcoef<-as.data.frame(MOMBcoef)
icinf<-t(MOMBcoef[,2]-1.96%*%MOMBcoef[,3])
icsup<-t(MOMBcoef[,2]+1.96%*%MOMBcoef[,3])
MOMBcoef<- cbind(MOMBcoef[,1:3],icinf,icsup,t(MOMB$medidas))
colnames(MOMBcoef)=c("parameter", "est", "std.error", "ci_inf","ci_sup", "AIC", "BIC"," AICc", "CAIC")


Gompertz = modelos(modelo = 'Gompertz',k=k)
Gompertz<-tidy(Gompertz$EMV)
Gompecoef<-as.data.frame(Gompertz)
icinf<-t(Gompertz[,2]-1.96%*%Gompertz[,3])
icsup<-t(Gompertz[,2]+1.96%*%Gompertz[,3])
Gompecoef<- cbind(Gompecoef[,1:3],icinf,icsup,t(Gompertz$medidas))
colnames(Gompecoef)=c("parameter", "est", "std.error", "ci_inf","ci_sup", "AIC", "BIC"," AICc", "CAIC")

Tabela_final = rbind(MOMBcoef,Gompecoef)
MO_Exp = modelos(modelo = 'MO-Exp', k=k)
MO_Exp<-tidy(MO_Exp$EMV)
MO_Expcoef<-as.data.frame(MO_Exp)
icinf<-t(MO_Exp[,2]-1.96%*%MO_Exp[,3])
icsup<-t(MO_Exp[,2]+1.96%*%MO_Exp[,3])
MO_Expcoef<- cbind(MO_Expcoef[,1:3],icinf,icsup,t(MO_Exp$medidas))
colnames(MO_Expcoef)=c("parameter", "est", "std.error", "ci_inf","ci_sup", "AIC", "BIC"," AICc", "CAIC")
Tabela_final = rbind(MOMBcoef,Gompecoef,MO_Expcoef)


Inverse_Gaussian = modelos(modelo = 'Inverse Gaussian',k=k)
Inverse_Gaussian$medidas
Inverse_Gaussian<-tidy(Inverse_Gaussian$EMV)
Inverse_Gaussiancoef<-as.data.frame(Inverse_Gaussian)
icinf<-t(Inverse_Gaussian[,2]-1.96%*%Inverse_Gaussian[,3])
icsup<-t(Inverse_Gaussian[,2]+1.96%*%Inverse_Gaussian[,3])
Inverse_Gaussiancoef<- cbind(Inverse_Gaussiancoef[,1:3],icinf,icsup,t(Inverse_Gaussian$medidas))
colnames(Inverse_Gaussiancoef)=c("parameter", "est", "std.error", "ci_inf","ci_sup", "AIC", "BIC"," AICc", "CAIC")
Tabela_final = rbind(MOMBcoef,Gompecoef,MO_Expcoef,Inverse_Gaussiancoef)


Stand_Mix_Exp = modelos(modelo = 'Standard Mixture Exponential', k=k ) # It's not working!!!!
Stand_Mix_Exp$medidas
Stand_Mix_Exp<-tidy(Stand_Mix_Exp$EMV)
Stand_Mix_Expcoef<-as.data.frame(Stand_Mix_Exp)
icinf<-t(Stand_Mix_Exp[,2]-1.96%*%Stand_Mix_Exp[,3])
icsup<-t(Stand_Mix_Exp[,2]+1.96%*%Stand_Mix_Exp[,3])
Stand_Mix_Expcoef<- cbind(Stand_Mix_Expcoef[,1:3],icinf,icsup,t(Stand_Mix_Exp$medidas))
colnames(Stand_Mix_Expcoef)=c("parameter", "est", "std.error", "ci_inf","ci_sup", "AIC", "BIC"," AICc", "CAIC")
Tabela_final = rbind(MOMBcoef,Gompecoef,MO_Expcoef,Inverse_Gaussiancoef,Stand_Mix_Expcoef)

dist<-c("MOMB", "", "Gompertz", "", "MO_Exp","", "Inv_Gaussian", "",  "stand_Mix","")
dist<-as.vector(dist)
Tabela_final2<-cbind(dist,Tabela_final)
Tabela_final2<-as.data.frame(Tabela_final2)
Tabela_final2[3:10]<- round(Tabela_final2[3:10],2)

Tabela_final2[2,7:10] = "";Tabela_final2[4,7:10] = "";Tabela_final2[6,7:10] = "";Tabela_final2[8,7:10] = "";Tabela_final2[10,7:10] = ""

Tabela_final2

t = seq(0, 10, 0.01)

modelo = 'Standard Mixture Exponential'
if(modelo == 'Standard Mixture Exponential'){
  acumulada = function(x, p, q) {
    p- p*exp(-q*x)
  }
}
st.sme = 1-acumulada(t, Stand_Mix_Exp$EMV@coef[1], Stand_Mix_Exp$EMV@coef[2])

modelo = 'Promotion Time Exponential'
if(modelo == 'Promotion Time Exponential'){
  acumulada = function(x, p, a) {
    G = pexp(x, a)
    return(1 - exp(-p*G))
  }
} 
st.pte = 1-acumulada(t, Prom_Time_Exp$EMV@coef[1], Prom_Time_Exp$EMV@coef[2])

modelo = 'MO-Exp'
if(modelo == 'MO-Exp'){
  acumulada = function(x, r, a) {
    G = pexp(x, a)
    return(1 - r*(1 - G)/(1 - (1 - r)*(1 - G)))
  } 
}
st.moexp = 1-acumulada(t, MO_Exp$EMV@coef[1], MO_Exp$EMV@coef[2])

modelo = 'Gompertz'
if(modelo == 'Gompertz'){
  acumulada = function(x, a, b) 1 - exp(-(exp(a*x) - 1)*b/a)
  # acumulada =function(x, a, b) 1 - exp(-b*(exp(a*x)-1))
}
st.gomp = 1-acumulada(t, Gompertz$EMV@coef[1], Gompertz$EMV@coef[2])

modelo = 'Inverse Gaussian'
if(modelo == 'Inverse Gaussian'){
  acumulada = function(x, a, b)  pnorm(sqrt(b/x)*(x/a - 1)) + exp(2*a/b)*pnorm(-sqrt(b/x)*(x/a + 1))  
}
st.invgauss = 1-acumulada(t, Inverse_Gaussian$EMV@coef[1], Inverse_Gaussian$EMV@coef[2])

modelo = 'MOMB'
if(modelo == 'MOMB'){
  acumulada_base = function(x, a) erf(x/(sqrt(2)*a))-(x*exp(-(x^2)/(2*a^2))*sqrt(2/(pi)))/a
  acumulada = function(x, a, b) ((acumulada_base(x = x, a = a))/((1-(1-b)*(1-acumulada_base(x = x, a = a))))) 
}
st.momb = 1-acumulada(t, MOMB$EMV@coef[1], MOMB$EMV@coef[2])

#plot

ekm = survfit(Surv(tempos, cens) ~ 1, se.fit = FALSE)

cores.google = c(blue = '#4285F4', red = '#DB4437', yellow = '#F4B400',
                 green = '#0F9D58')


pdf("rplotexemplo1.pdf", width = 100, height = 50) 

par(mfrow=c(2,2))
plot(ekm, lwd = 2)
lines(t, st.momb, type = 'l', col = cores.google[2], ylim = c(0, 1), lwd = 2)
lines(t, st.gomp, type = 'l', col = cores.google[1], ylim = c(0, 1), lwd = 2)

legend(x = 'topright', legend = c(paste('MOMB','AIC:', round(MOMB$AIC,2), sep = ' '),
                                  paste('Gompertz','AIC:', round(Gompertz$AIC,2), sep = ' ')#,
                                  ,paste('Kaplan Meier')),col=c('navy',cores.google[1]),
       text.col = c(cores.google[2], cores.google[1],'black'), cex = 1)

plot(ekm, lwd = 2)
lines(t, st.momb, type = 'l', col = cores.google[2], ylim = c(0, 1), lwd = 2)
lines(t, st.invgauss, type = 'l', col = cores.google[1], ylim = c(0, 1), lwd = 2)


legend(x = 'topright',legend = c(paste('MOMB','AIC:', round(MOMB$AIC,2), sep = ' '),
                                 paste('Inverse Gaussian','AIC:', round(Inverse_Gaussian$AIC,2), sep = ' ')
                                 ,paste('Kaplan Meier')),col=c(cores.google[2],  
                                                               cores.google[2],'black'),
       text.col = c(cores.google[2] , cores.google[1],'black'), cex = 1)


plot(ekm, lwd = 2)
lines(t, st.momb, type = 'l', col = cores.google[2], ylim = c(0, 1), lwd = 2)
lines(t, st.moexp, type = 'l', col = cores.google[1], ylim = c(0, 1), lwd = 2)

legend(x = 'topright',legend = c(paste('MOMB','AIC:', round(MOMB$AIC,2), sep = ' '),
                                 paste('Marshall Olkin Exponential','AIC:', round(MO_Exp$AIC,2), sep = ' ')
                                 ,paste('Kaplan Meier')),col=c(cores.google[2],  
                                                               cores.google[2],'black'),
       text.col = c(cores.google[2] , cores.google[1],'black'), cex = 1)

plot(ekm, lwd = 2)
lines(t, st.momb, type = 'l', col = cores.google[2] , ylim = c(0, 1), lwd = 2)
lines(t, st.pte, type = 'l', col = cores.google[1], ylim = c(0, 1), lwd = 2)

legend(x = 'topright',legend = c(paste('MOMB','AIC:', round(MOMB$AIC,2), sep = ' '),
                                 paste('Promotion Time Exponential','AIC:', round(Prom_Time_Exp$AIC,2), sep = ' ')
                                 ,paste('Kaplan Meier')),col=c(cores.google[2],  
                                                               cores.google[2],'black'),
       text.col = c(cores.google[2] , cores.google[1],'black'), cex = 1)

plot(ekm, lwd = 2)
lines(t, st.momb, type = 'l', col = cores.google[2] , ylim = c(0, 1), lwd = 2)
lines(t, st.sme, type = 'l', col = cores.google[1], ylim = c(0, 1), lwd = 2)

legend(x = 'topright',legend = c(paste('MOMB','AIC:', round(MOMB$AIC,2), sep = ' '),
                                 paste('Standard Mixture Exponential','AIC:', round(Stand_Mix_Exp$AIC,2), sep = ' ') ,paste('Kaplan Meier')
),col=c(cores.google[2],  
        cores.google[2],'black'),
text.col = c(cores.google[2] , cores.google[1],'black'), cex = 1)





###################### Aplica??o 2 ###################
#pacotes utilizados
library(survival)
library(bbmle)
library(msm)
library(VGAM) #funcao 'erf'
library(broom) 
#carregando dados
# OK Times to death (in months) for breast cancer patients with different immunohistochemical responses, Immunoperoxidase Negative

t1=c(19,25,30,34,37,46,47,51,56,57,61,66,67,74,78,86,122,123,130,130,133,134,136,141,143,148,151,152,153,154,156,162,164,165,182,189)
c1=c(rep(1,16),rep(0,20))


set.seed(2019)
# Distribution's parameters estimation

modelos = function (varnum, modelo = 'MOMB',k=k, cov_a = FALSE, cov_b = FALSE){
  tempos = t1/max(t1)
  cens = c1
  
  
  if(modelo == 'Standard Mixture Exponential'){
    densidade = function(x, p, q){
      p*q*exp(-q*x)}
    acumulada = function(x, p, q) {
      p- p*exp(-q*x)
    }
  }
  
  
  if(modelo == 'MO-Exp'){
    densidade = function(x, r, a){
      f = dexp(x, a)
      G = pexp(x, a)
      return(r*f/((1 - (1 - r)*(1 - G))^2))
    }
    acumulada = function(x, r, a) {
      G = pexp(x, a)
      return(1- r*(1-G)/(1 - (1 - r)*(1 - G)))
    } 
  }
  
  if(modelo == 'Gompertz'){
    # densidade = function(x, a, b) b*exp(a*x)*exp(-(exp(a*x) - 1)*b/a)
    densidade = function(x, a, b) a*b*exp(b)*exp(a*x)*exp(-b*exp(a*x))
    #acumulada = function(x, a, b) 1 - exp(-(exp(a*x) - 1)*b/a) 
    acumulada = function(x, a, b) 1 - exp(-b*(exp(a*x)-1))
  }
  
  if(modelo == 'Inverse Gaussian'){
    #densidade = function(x, a, b) exp(-(1 - a*x)^2/(2*b*x))/(sqrt(2*b*pi*x^3))  
    #acumulada = function(x, a, b) pnorm((-1+a*x)/(sqrt(b*x)))+exp(2*a/b)*pnorm((-1-a*x)/(sqrt(b*x)))
    densidade = function(x, a, b) exp(-b*(x-a)^2/(2*a^2*x))*(b/(2*pi*x^3))^(1/2)  
    acumulada = function(x, a, b) pnorm(sqrt(b/x)*(x/a - 1)) + exp(2*a/b)*pnorm(-sqrt(b/x)*(x/a + 1))  
  }
  
  if(modelo == 'MOMB'){
    densidade_base = function(x, a) a^(-3)*x^2*exp(-x^2/(2*a^2))*sqrt(2/pi)
    acumulada_base = function(x, a) erf(x/(sqrt(2)*a))-(x*exp(-(x^2)/(2*a^2))*sqrt(2/(pi)))/a
    
    densidade = function(x, a, b) b*densidade_base(x = x, a = a)/((1-(1-b)*(1-acumulada_base(x = x, a = a)))^2)
    #acumulada = function(x, a, b) 1-((b*(1-acumulada_base(x = x, a = a)))/((1-(1-b)*(1-acumulada_base(x = x, a = a))))) 
    acumulada = function(x, a, b) ((acumulada_base(x = x, a = a))/((1-(1-b)*(1-acumulada_base(x = x, a = a))))) 
  }
  
  # Fun??o de logverossimilhan?a sem as covari?veis nos par?metros
  loglike = function(a ,b){         # tempos e 'cens' = censuras
    f = densidade(tempos, a ,b)
    s = 1 - acumulada(tempos, a ,b)
    logl = sum((cens*log(f)) + ((1 - cens)*log(s)))
    return(-logl)
  } 
  
  #Estimador de Maxima Verossimilhan?a
  library(bbmle)
  st = list(a = runif(1), b = runif(1))
  emv = mle2(loglike, start = st, method = 'BFGS')
  a=AIC(emv)
  aic= -2*(logLik(emv)) +2*(2)
  AICc=-2*(logLik(emv)) +2*(2)+ 2*(2*(2+1))/(length( t1)-2-1)
  bic= -2*(logLik(emv))+2*log(length( t1))
  CAIC= 0.5 * aic + (1 - 0.5) * bic 
  return(list(AIC = a,medidas = c(aic,bic,CAIC,AICc), EMV = emv))
}


tempos = t1/max(t1)
cens = c1

MOMB1 = modelos(modelo = 'MOMB', k=k)
MOMB1$medidas->medidas
MOMBcoef<-tidy(MOMB1$EMV);MOMBcoef
MOMBcoef<-as.data.frame(MOMBcoef)
icinf<-t(MOMBcoef[,2]-1.96%*%MOMBcoef[,3])
icsup<-t(MOMBcoef[,2]+1.96%*%MOMBcoef[,3])
MOMBcoef<- cbind(MOMBcoef[,1:3],icinf,icsup,t(medidas))
p<-c("p","","","","","","","","")
colnames(MOMBcoef)=c("parameter", "est", "std.error", "ci_inf","ci_sup", "AIC", "BIC"," AICc", "CAIC")
MOMBcoef[,2:9]<- round(MOMBcoef[,2:9],2)
MOMBcoef<-rbind(MOMBcoef,p)

Gompertz1 = modelos(modelo = 'Gompertz',k=k)
medidas<-Gompertz1$medidas
Gompertz<-tidy(Gompertz1$EMV)
Gompecoef<-as.data.frame(Gompertz)
icinf<-t(Gompertz[,2]-1.96%*%Gompertz[,3])
icsup<-t(Gompertz[,2]+1.96%*%Gompertz[,3])
Gompecoef<- cbind(Gompecoef[,1:3],icinf,icsup,t(medidas))
colnames(Gompecoef)=c("parameter", "est", "std.error", "ci_inf","ci_sup", "AIC", "BIC"," AICc", "CAIC")
Gompecoef[,2:9]<- round(Gompecoef[,2:9],2)
Gompecoef<-rbind(Gompecoef,p)
Tabela_final = rbind(MOMBcoef,Gompecoef)

MO_Exp1 = modelos(modelo = 'MO-Exp', k=k)
MO_Exp1$medidas->medidas
MO_Exp<-tidy(MO_Exp1$EMV)
MO_Expcoef<-as.data.frame(MO_Exp)
icinf<-t(MO_Exp[,2]-1.96%*%MO_Exp[,3])
icsup<-t(MO_Exp[,2]+1.96%*%MO_Exp[,3])
MO_Expcoef<- cbind(MO_Expcoef[,1:3],icinf,icsup,t(medidas))
colnames(MO_Expcoef)=c("parameter", "est", "std.error", "ci_inf","ci_sup", "AIC", "BIC"," AICc", "CAIC")
MO_Expcoef[,2:9]<- round(MO_Expcoef[,2:9],2)

MO_Expcoef<-rbind(MO_Expcoef,p)

Tabela_final = rbind(MOMBcoef,Gompecoef,MO_Expcoef)


Inverse_Gaussian1 = modelos(modelo = 'Inverse Gaussian',k=k)
Inverse_Gaussian1$medidas -> medidas
Inverse_Gaussian<-tidy(Inverse_Gaussian1$EMV)
Inverse_Gaussiancoef<-as.data.frame(Inverse_Gaussian)
icinf<-t(Inverse_Gaussian[,2]-1.96%*%Inverse_Gaussian[,3])
icsup<-t(Inverse_Gaussian[,2]+1.96%*%Inverse_Gaussian[,3])
Inverse_Gaussiancoef<- cbind(Inverse_Gaussiancoef[,1:3],icinf,icsup,t(medidas))
colnames(Inverse_Gaussiancoef)=c("parameter", "est", "std.error", "ci_inf","ci_sup", "AIC", "BIC"," AICc", "CAIC")
Inverse_Gaussiancoef[,2:9]<- round(Inverse_Gaussiancoef[,2:9],2)

Inverse_Gaussiancoef<-rbind(Inverse_Gaussiancoef,p)

Tabela_final = rbind(MOMBcoef,Gompecoef,MO_Expcoef,Inverse_Gaussiancoef)


Stand_Mix_Exp1 = modelos(modelo = 'Standard Mixture Exponential', k=k ) # It's not working!!!!
Stand_Mix_Exp1$medidas->medidas
Stand_Mix_Exp<-tidy(Stand_Mix_Exp1$EMV)
Stand_Mix_Expcoef<-as.data.frame(Stand_Mix_Exp)
icinf<-t(Stand_Mix_Exp[,2]-1.96%*%Stand_Mix_Exp[,3])
icsup<-t(Stand_Mix_Exp[,2]+1.96%*%Stand_Mix_Exp[,3])
Stand_Mix_Expcoef<- cbind(Stand_Mix_Expcoef[,1:3],icinf,icsup,t(medidas))
colnames(Stand_Mix_Expcoef)=c("parameter", "est", "std.error", "ci_inf","ci_sup", "AIC", "BIC"," AICc", "CAIC")
Stand_Mix_Expcoef[,2:9]<- round(Stand_Mix_Expcoef[,2:9],2)

Stand_Mix_Expcoef<-rbind(Stand_Mix_Expcoef,p)
Tabela_final = rbind(MOMBcoef,Gompecoef,MO_Expcoef,Inverse_Gaussiancoef,Stand_Mix_Expcoef)

dist<-c("MOMB", "", "", "Gompertz", "","",  "MO_Exp","", "", "Inv_Gaussian", "", "",  "stand_Mix","", "")
dist<-as.vector(dist)
Tabela_final2<-cbind(dist,Tabela_final)
Tabela_final2<-as.data.frame(Tabela_final2)

Tabela_final2[2,7:10] = "";Tabela_final2[5,7:10] = "";Tabela_final2[8,7:10] = "";Tabela_final2[11,7:10] = "";Tabela_final2[14,7:10] = ""

Tabela_final2



t = seq(0, 10, 0.01)

modelo = 'Standard Mixture Exponential'
if(modelo == 'Standard Mixture Exponential'){
  acumuladamix = function(x, p, q) {
    p- p*exp(-q*x)
  }
}
st.sme = 1-acumuladamix(t, Stand_Mix_Exp1$EMV@coef[1], Stand_Mix_Exp1$EMV@coef[2])


modelo = 'MO-Exp'
if(modelo == 'MO-Exp'){
  acumuladamo = function(x, r, a) {
    G = pexp(x, a)
    return(1- r*(1-G)/(1 - (1 - r)*(1 - G)))
  } 
}
st.moexp = 1-acumuladamo(t, MO_Exp1$EMV@coef[1], MO_Exp1$EMV@coef[2])

modelo = 'Gompertz'
if(modelo == 'Gompertz'){
  #acumulada = function(x, a, b) 1 - exp(-(exp(a*x) - 1)*b/a) 
  acumuladagom =function(x, a, b) 1 - exp(-b*(exp(a*x)-1))
}
st.gomp = 1-acumuladagom(t, Gompertz1$EMV@coef[1], Gompertz1$EMV@coef[2])

modelo = 'Inverse Gaussian'
if(modelo == 'Inverse Gaussian'){
  acumuladaiv = function(x, a, b)  pnorm(sqrt(b/x)*(x/a - 1)) + exp(2*a/b)*pnorm(-sqrt(b/x)*(x/a + 1))  
}
st.invgauss = 1-acumuladaiv(t, Inverse_Gaussian1$EMV@coef[1], Inverse_Gaussian1$EMV@coef[2])

modelo = 'MOMB'
if(modelo == 'MOMB'){
  acumulada_base = function(x, a) erf(x/(sqrt(2)*a))-(x*exp(-(x^2)/(2*a^2))*sqrt(2/(pi)))/a
  acumuladamomb = function(x, a, b) ((acumulada_base(x = x, a = a))/((1-(1-b)*(1-acumulada_base(x = x, a = a))))) 
}
st.momb = 1-acumuladamomb(t, MOMB1$EMV@coef[1], MOMB1$EMV@coef[2])

#plot

ekm = survfit(Surv(tempos, cens) ~ 1, se.fit = FALSE)

cores.google = c(blue = '#4285F4', red = '#DB4437', yellow = '#F4B400',
                 green = '#0F9D58')


st<-ekm$surv

pdf("rplotxemplo2.pdf", width = 12, height = 8) 
par(mfrow=c(2,2))
plot(ekm, lwd = 2)
lines(t, st.momb, type = 'l', col = cores.google[2], ylim = c(0, 1), lwd = 2)
lines(t, st.gomp, type = 'l', col = cores.google[1], ylim = c(0, 1), lwd = 2)

legend(x = 'topright', legend = c(paste('MOMB','AIC:', round(MOMB$AIC,2), sep = ' '),
                                  paste('Gompertz','AIC:', round(Gompertz$AIC,2), sep = ' ')#,
                                  ,paste('Kaplan Meier')),col=c('navy',cores.google[1]),
       text.col = c(cores.google[2], cores.google[1],'black'), cex = 1)

plot(ekm, lwd = 2)
lines(t, st.momb, type = 'l', col = cores.google[2], ylim = c(0, 1), lwd = 2)
lines(t, st.invgauss, type = 'l', col = cores.google[1], ylim = c(0, 1), lwd = 2)


legend(x = 'topright',legend = c(paste('MOMB','AIC:', round(MOMB$AIC,2), sep = ' '),
                                 paste('Inverse Gaussian','AIC:', round(Inverse_Gaussian$AIC,2), sep = ' ')
                                 ,paste('Kaplan Meier')),col=c(cores.google[2],  
                                                               cores.google[2],'black'),
       text.col = c(cores.google[2] , cores.google[1],'black'), cex = 1)


plot(ekm, lwd = 2)
lines(t, st.momb, type = 'l', col = cores.google[2], ylim = c(0, 1), lwd = 2)
lines(t, st.moexp, type = 'l', col = cores.google[1], ylim = c(0, 1), lwd = 2)

legend(x = 'topright',legend = c(paste('MOMB','AIC:', round(MOMB$AIC,2), sep = ' '),
                                 paste('Marshall Olkin Exponential','AIC:', round(MO_Exp$AIC,2), sep = ' ')
                                 ,paste('Kaplan Meier')),col=c(cores.google[2],  
                                                               cores.google[2],'black'),
       text.col = c(cores.google[2] , cores.google[1],'black'), cex = 1)

#plot(ekm, lwd = 2)
#lines(t, st.momb, type = 'l', col = cores.google[2] , ylim = c(0, 1), lwd = 2)
#lines(t, st.pte, type = 'l', col = cores.google[1], ylim = c(0, 1), lwd = 2)

#legend(x = 'topright',legend = c(paste('MOMB','AIC:', round(MOMB$AIC,2), sep = ' '),
#                                paste('Promotion Time Exponential','AIC:', round(Prom_Time_Exp$AIC,2), sep = ' ')
#                                ,paste('Kaplan Meier')),col=c(cores.google[2],  
#                                                              cores.google[2],'black'),
#       text.col = c(cores.google[2] , cores.google[1],'black'), cex = 1)

plot(ekm, lwd = 2)
lines(t, st.momb, type = 'l', col = cores.google[2] , ylim = c(0, 1), lwd = 2)
lines(t, st.sme, type = 'l', col = cores.google[1], ylim = c(0, 1), lwd = 2)

legend(x = 'topright',legend = c(paste('MOMB','AIC:', round(MOMB$AIC,2), sep = ' '),
                                 paste('Standard Mixture Exponential','AIC:', round(Stand_Mix_Exp$AIC,2), sep = ' ') ,paste('Kaplan Meier')
),col=c(cores.google[2],  
        cores.google[2],'black'),
text.col = c(cores.google[2] , cores.google[1],'black'), cex = 1)





dev.off() 






