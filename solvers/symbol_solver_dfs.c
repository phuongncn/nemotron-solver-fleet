// SOLVER_VERSION=v6.2-doubt-nocap
/*
 * Symbol cipher solver v6 — C. (examples-only DOUBT mode + 90s cap)
 * Fixes over v4:
 * - BUG FIX: n_arith==0 no longer skips DFS (concat-only examples + arithmetic query)
 * - NEW: Query equation used as DFS constraint (improves pruning for query-only ops)
 * - NEW: Additional operations: AND, OR, GCD
 * - NEW: '-' treated as digit when not an operator (dash_as_digit mode)
 * - Preserved: smart symbol ordering, output-driven pruning, operator-as-neg-sign
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXEX 10
#define MAXSYM 20
#define MAX_COMBOS 4096

// Puzzle data
char pid[64];
int n_ex;
int ex_l0[MAXEX], ex_l1[MAXEX], ex_r0[MAXEX], ex_r1[MAXEX], ex_op[MAXEX];
char ex_out[MAXEX][8];
int ex_olen[MAXEX];
int q_l0, q_l1, q_r0, q_r1, q_op;
char answer[8];
int ans_len;
int syms[MAXSYM], n_sym;
int sym_order[MAXSYM]; // optimized assignment order

// DFS state
int m[128], inv[32], used[32];
int base_g;
int opassign[128];
int is_le;

// Timeout
time_t puzzle_start;
int timed_out;
int puzzle_timeout;
long long dfs_nodes;

// Constraint tracking: which equations involve each symbol
int sym_in_eq[128][MAXEX]; // sym_in_eq[sym][i] = 1 if sym appears in equation i
int sym_eq_count[128];     // how many equations reference this symbol

void init_state() {
    memset(m, 0xff, sizeof(m));
    memset(inv, 0xff, sizeof(inv));
    memset(used, 0, sizeof(used));
}

int to_val(int c0, int c1) {
    if (is_le) return m[c0] + m[c1] * base_g;
    return m[c0] * base_g + m[c1];
}

int gcd(int a, int b) { while(b){int t=b;b=a%b;a=t;} return a; }

int compute(int lv, int rv, int ot) {
    if (ot == '+') return lv + rv;
    if (ot == '-') return lv - rv;
    if (ot == 'r') return rv - lv;                                     // reverse subtract
    if (ot == '*') return lv * rv;
    if (ot == '%') return rv != 0 ? lv % rv : -999999;                // a % b
    if (ot == 'M') { int mx=lv>rv?lv:rv, mn=lv<rv?lv:rv; return mn>0 ? mx%mn : -999999; } // max%min
    if (ot == '/') return rv != 0 ? lv / rv : -999999;                // a / b
    if (ot == 'D') { int mx=lv>rv?lv:rv, mn=lv<rv?lv:rv; return mn>0 ? mx/mn : -999999; } // max/min
    if (ot == 'A') return lv > rv ? lv - rv : rv - lv;                // abs diff
    if (ot == 'X') return lv ^ rv;                                     // XOR
    if (ot == '&') return lv & rv;                                     // AND
    if (ot == 'O') return lv | rv;                                     // OR
    if (ot == 'G') { int a=lv>0?lv:-lv, b=rv>0?rv:-rv; return (a==0&&b==0)?0:gcd(a?a:b,b?b:a); } // GCD
    return -999999;
}

int check_output(int val, const char *out, int olen);

int output_offset = 0;  // global: 0=normal, ±1, ±2
int reverse_output = 0; // global: 0=normal, 1=reverse output digits

// --- examples-only / DOUBT mode (hypothesis: official answer may be wrong) ---
int examples_only = 0;       // 1=find any mapping fitting examples; 2=find one with query-pred != g_avoid
char g_pred[64];             // predicted query output for the found mapping
char g_map[256];             // symbol=digit mapping of the found mapping (for DOUBT verification)
char g_avoid[64];            // (mode 2) reject mappings whose query-pred equals this
int q_op_sym = 0;            // the query operator's source symbol char (for sign prefix)

// Encode the query result `val` (post-offset applied here) into output symbols of length olen.
// Mirrors check_output_neg + check_output but EMITS instead of comparing. Returns 1 on success.
int encode_pred(int val, int olen, char *buf) {
    val += output_offset;
    int start = 0;
    if (val < 0) {
        if (olen < 2) return 0;
        buf[0] = (char)q_op_sym;   // sign symbol = query operator (mirror check_output_neg)
        start = 1; olen -= 1; val = -val;
    }
    int d[8], nd = 0;
    if (val == 0) d[nd++] = 0;
    else { int t = val; while (t > 0) { d[nd++] = t % base_g; t /= base_g; } }
    // d[] is little-endian digit order here
    if (is_le) { while (nd < olen) d[nd++] = 0; }
    else {
        for (int i = 0; i < nd/2; i++){int t=d[i];d[i]=d[nd-1-i];d[nd-1-i]=t;}
        if (nd < olen){ for(int i=nd;i>0;i--) d[i+olen-nd-1]=d[i-1]; for(int i=0;i<olen-nd;i++) d[i]=0; nd=olen; }
    }
    if (nd != olen) return 0;
    if (reverse_output){ for(int i=0;i<nd/2;i++){int t=d[i];d[i]=d[nd-1-i];d[nd-1-i]=t;} }
    for (int j = 0; j < nd; j++){ if (d[j] >= base_g || inv[d[j]] < 0) return 0; buf[start+j] = (char)inv[d[j]]; }
    buf[start+nd] = 0;
    return 1;
}

int check_output_neg(int val, const char *out, int olen) {
    // Apply offset BEFORE sign check
    val += output_offset;
    if (val < 0 && olen >= 2 && out[0] == '-') {
        return check_output(-val, out + 1, olen - 1);
    }
    if (val < 0 && olen >= 2) {
        int fc = (unsigned char)out[0];
        if (opassign[fc] != 0) {
            return check_output(-val, out + 1, olen - 1);
        }
    }
    if (val >= 0) return check_output(val, out, olen);
    return 0;
}

int check_output(int val, const char *out, int olen) {
    // Note: offset already applied by check_output_neg caller
    if (val < 0) return 0;
    int d[8], nd = 0;
    if (val == 0) { d[nd++] = 0; }
    else { int t = val; while (t > 0) { d[nd++] = t % base_g; t /= base_g; } }

    if (is_le) {
        while (nd < olen) d[nd++] = 0;
    } else {
        for (int i = 0; i < nd/2; i++) { int t = d[i]; d[i] = d[nd-1-i]; d[nd-1-i] = t; }
        if (nd < olen) {
            for (int i = nd; i > 0; i--) d[i + olen - nd - 1] = d[i-1];
            for (int i = 0; i < olen - nd; i++) d[i] = 0;
            nd = olen;
        }
    }
    if (nd != olen) return 0;

    if (reverse_output) {
        for (int i = 0; i < nd/2; i++) { int t = d[i]; d[i] = d[nd-1-i]; d[nd-1-i] = t; }
    }

    for (int j = 0; j < nd; j++) {
        if (d[j] >= base_g) return 0;
        if (inv[d[j]] != (int)(unsigned char)out[j]) return 0;
    }
    return 1;
}

/* Enhanced partial check: checks as much as possible with current assignments */
int check_partial_enhanced(int ei) {
    int ot = opassign[ex_op[ei]];
    if (ot == 'C' || ot == 'R') return 1;

    int l0 = m[ex_l0[ei]], l1 = m[ex_l1[ei]], r0 = m[ex_r0[ei]], r1 = m[ex_r1[ei]];

    // If all 4 operand digits known, compute and check output
    if (l0 >= 0 && l1 >= 0 && r0 >= 0 && r1 >= 0) {
        int lv = is_le ? l0 + l1 * base_g : l0 * base_g + l1;
        int rv = is_le ? r0 + r1 * base_g : r0 * base_g + r1;
        int val = compute(lv, rv, ot);
        val += output_offset;  // apply offset BEFORE sign check
        int olen = ex_olen[ei];

        // Handle negative
        int start = 0;
        if (val < 0 && olen >= 2) {
            int fc = (unsigned char)ex_out[ei][0];
            if (fc == '-' || opassign[fc] != 0) {
                val = -val;
                start = 1;
                olen--;
            } else {
                return 0;
            }
        }
        if (val < 0) return 0;

        // Encode and check against output
        int d[8], nd = 0;
        if (val == 0) { d[nd++] = 0; }
        else { int t = val; while (t > 0) { d[nd++] = t % base_g; t /= base_g; } }

        if (is_le) {
            while (nd < olen) d[nd++] = 0;
        } else {
            for (int i = 0; i < nd/2; i++) { int t = d[i]; d[i] = d[nd-1-i]; d[nd-1-i] = t; }
            if (nd < olen) {
                for (int i = nd; i > 0; i--) d[i + olen - nd - 1] = d[i-1];
                for (int i = 0; i < olen - nd; i++) d[i] = 0;
                nd = olen;
            }
        }
        if (nd != olen) return 0;

        if (reverse_output) {
            for (int i = 0; i < nd/2; i++) { int t = d[i]; d[i] = d[nd-1-i]; d[nd-1-i] = t; }
        }

        for (int j = 0; j < nd; j++) {
            if (d[j] >= base_g) return 0;
            int oc = (int)(unsigned char)ex_out[ei][j + start];
            if (m[oc] >= 0 && m[oc] != d[j]) return 0;
            if (m[oc] < 0 && used[d[j]] && inv[d[j]] != oc) return 0;
        }
        return 1;
    }

    // Partial: if we know some output digits, check consistency
    // E.g., if output char is already mapped, check digit range
    int olen = ex_olen[ei];
    int start = 0;
    if (olen >= 2) {
        int fc = (unsigned char)ex_out[ei][0];
        if (fc == '-' || (opassign[fc] != 0 && m[fc] < 0)) {
            start = 1;
        }
    }
    for (int j = start; j < ex_olen[ei]; j++) {
        int oc = (int)(unsigned char)ex_out[ei][j];
        if (m[oc] >= 0 && m[oc] >= base_g) return 0;
    }

    return 1;
}

int n_arith;
int arith_idx[MAXEX];

int dfs(int idx) {
    if (idx == n_sym) {
        // Full verify
        for (int i = 0; i < n_arith; i++) {
            int ei = arith_idx[i];
            int lv = to_val(ex_l0[ei], ex_l1[ei]);
            int rv = to_val(ex_r0[ei], ex_r1[ei]);
            int val = compute(lv, rv, opassign[ex_op[ei]]);
            if (!check_output_neg(val, ex_out[ei], ex_olen[ei])) return 0;
        }
        int qot = opassign[q_op];
        if (examples_only) {
            // Examples already verified above. Produce the query prediction g_pred.
            char pred[64];
            if (qot == 'C') { pred[0]=q_l0;pred[1]=q_l1;pred[2]=q_r0;pred[3]=q_r1;pred[4]=0; }
            else if (qot == 'R') { pred[0]=q_r0;pred[1]=q_r1;pred[2]=q_l0;pred[3]=q_l1;pred[4]=0; }
            else {
                int lv = to_val(q_l0, q_l1), rv = to_val(q_r0, q_r1);
                if (!encode_pred(compute(lv, rv, qot), ans_len, pred)) return 0;
            }
            if (examples_only == 2 && strcmp(pred, g_avoid) == 0) return 0; // need a DIFFERENT pred
            strncpy(g_pred, pred, sizeof(g_pred)-1);
            { int gp=0; g_map[0]=0;
              for (int i=0;i<n_sym;i++) gp+=snprintf(g_map+gp, sizeof(g_map)-gp, "%s%c=%d", i?",":"", sym_order[i], m[sym_order[i]]); }
            return 1;
        }
        if (qot == 'C') {
            char p[5]; p[0]=q_l0; p[1]=q_l1; p[2]=q_r0; p[3]=q_r1; p[4]=0;
            return ans_len == 4 && memcmp(p, answer, 4) == 0;
        }
        if (qot == 'R') {
            char p[5]; p[0]=q_r0; p[1]=q_r1; p[2]=q_l0; p[3]=q_l1; p[4]=0;
            return ans_len == 4 && memcmp(p, answer, 4) == 0;
        }
        int lv = to_val(q_l0, q_l1);
        int rv = to_val(q_r0, q_r1);
        int val = compute(lv, rv, qot);
        return check_output_neg(val, answer, ans_len);
    }

    // Timeout check
    if (puzzle_timeout > 0 && ++dfs_nodes % 200000 == 0) {
        if (time(NULL) - puzzle_start >= puzzle_timeout) {
            timed_out = 1;
            return 0;
        }
    }

    int s = sym_order[idx]; // Use optimized order
    for (int d = 0; d < base_g; d++) {
        if (used[d]) continue;
        m[s] = d; inv[d] = s; used[d] = 1;

        // Enhanced pruning: check ALL equations this symbol participates in
        int prune = 0;
        for (int i = 0; i < n_arith && !prune; i++) {
            if (!check_partial_enhanced(arith_idx[i])) prune = 1;
        }
        // Also check query as constraint (early pruning for query-only ops)
        // — skip in examples_only mode (we don't constrain the query to the official answer)
        if (!prune && !examples_only && opassign[q_op] != 'C' && opassign[q_op] != 'R') {
            int ql0=m[q_l0], ql1=m[q_l1], qr0=m[q_r0], qr1=m[q_r1];
            if (ql0>=0 && ql1>=0 && qr0>=0 && qr1>=0) {
                // Check all answer symbols are assigned too
                int ans_ok = 1;
                for (int j = 0; j < ans_len; j++) {
                    int ac = (unsigned char)answer[j];
                    if (ac != '-' && m[ac] < 0) { ans_ok = 0; break; }
                }
                if (ans_ok) {
                    int lv = is_le ? ql0+ql1*base_g : ql0*base_g+ql1;
                    int rv = is_le ? qr0+qr1*base_g : qr0*base_g+qr1;
                    int val = compute(lv, rv, opassign[q_op]);
                    if (!check_output_neg(val, answer, ans_len)) prune = 1;
                }
            }
        }

        if (!prune && !timed_out && dfs(idx + 1)) return 1;
        if (timed_out) { m[s] = -1; inv[d] = -1; used[d] = 0; return 0; }

        m[s] = -1; inv[d] = -1; used[d] = 0;
    }
    return 0;
}

/* Compute optimal symbol ordering: most constrained first */
void compute_sym_order() {
    // Count how many equation operands/outputs reference each symbol
    memset(sym_eq_count, 0, sizeof(sym_eq_count));

    for (int i = 0; i < n_arith; i++) {
        int ei = arith_idx[i];
        sym_eq_count[ex_l0[ei]] += 2;
        sym_eq_count[ex_l1[ei]] += 2;
        sym_eq_count[ex_r0[ei]] += 2;
        sym_eq_count[ex_r1[ei]] += 2;
        // Output symbols get weight too
        for (int j = 0; j < ex_olen[ei]; j++) {
            int oc = (unsigned char)ex_out[ei][j];
            if (oc != '-') sym_eq_count[oc] += 1;
        }
    }
    // Query symbols (high weight — enables early query pruning)
    sym_eq_count[q_l0] += 2;
    sym_eq_count[q_l1] += 2;
    sym_eq_count[q_r0] += 2;
    sym_eq_count[q_r1] += 2;
    for (int j = 0; j < ans_len; j++) {
        int ac = (unsigned char)answer[j];
        if (ac != '-') sym_eq_count[ac] += 2;
    }

    // Sort syms by constraint count (descending)
    for (int i = 0; i < n_sym; i++) sym_order[i] = syms[i];

    // Simple selection sort (n_sym <= 20)
    for (int i = 0; i < n_sym - 1; i++) {
        int max_idx = i;
        for (int j = i + 1; j < n_sym; j++) {
            if (sym_eq_count[sym_order[j]] > sym_eq_count[sym_order[max_idx]])
                max_idx = j;
        }
        if (max_idx != i) {
            int tmp = sym_order[i];
            sym_order[i] = sym_order[max_idx];
            sym_order[max_idx] = tmp;
        }
    }
}

// Generate operator combos filtered by output length constraints
void gen_combos(int unique_ops[], int n_uops, int op_combos[][5], int *n_combos) {
    int is_concat[128]; memset(is_concat, 0, sizeof(is_concat));
    int is_revconcat[128]; memset(is_revconcat, 0, sizeof(is_revconcat));
    int has_examples[128]; memset(has_examples, 0, sizeof(has_examples));
    int max_olen[128]; memset(max_olen, 0, sizeof(max_olen));
    int min_olen[128]; memset(min_olen, 0x7f, sizeof(min_olen));
    int has_neg[128]; memset(has_neg, 0, sizeof(has_neg));

    for (int oi = 0; oi < n_uops; oi++) {
        int op = unique_ops[oi];
        int all_c = 1, all_r = 1, count = 0;
        for (int i = 0; i < n_ex; i++) {
            if (ex_op[i] != op) continue;
            count++;
            int olen = ex_olen[i];
            if (olen > max_olen[op]) max_olen[op] = olen;
            if (olen < min_olen[op]) min_olen[op] = olen;
            // Detect negative: starts with '-' or starts with any operator char
            int fc = (unsigned char)ex_out[i][0];
            if (fc == '-') has_neg[op] = 1;
            for (int oj = 0; oj < n_uops; oj++)
                if (fc == unique_ops[oj]) has_neg[op] = 1;
            char c[5]; c[0]=ex_l0[i]; c[1]=ex_l1[i]; c[2]=ex_r0[i]; c[3]=ex_r1[i]; c[4]=0;
            char r[5]; r[0]=ex_r0[i]; r[1]=ex_r1[i]; r[2]=ex_l0[i]; r[3]=ex_l1[i]; r[4]=0;
            if (olen != 4 || memcmp(c, ex_out[i], 4) != 0) all_c = 0;
            if (olen != 4 || memcmp(r, ex_out[i], 4) != 0) all_r = 0;
        }
        has_examples[op] = count;
        if (count > 0 && all_c) is_concat[op] = 1;
        if (count > 0 && all_r) is_revconcat[op] = 1;
    }

    *n_combos = 0;
    int op_cands[5][12];
    int n_cands[5];

    for (int oi = 0; oi < n_uops; oi++) {
        int op = unique_ops[oi];
        n_cands[oi] = 0;

        if (has_examples[op] == 0) {
            // Query-only operator: try all (can't filter without examples)
            op_cands[oi][n_cands[oi]++] = '+';
            op_cands[oi][n_cands[oi]++] = '-';
            op_cands[oi][n_cands[oi]++] = 'r';
            op_cands[oi][n_cands[oi]++] = '*';
            op_cands[oi][n_cands[oi]++] = 'M';
            op_cands[oi][n_cands[oi]++] = 'A';
            op_cands[oi][n_cands[oi]++] = '%';
            op_cands[oi][n_cands[oi]++] = 'D';
            op_cands[oi][n_cands[oi]++] = '/';
            op_cands[oi][n_cands[oi]++] = 'X';
            op_cands[oi][n_cands[oi]++] = '&';
            op_cands[oi][n_cands[oi]++] = 'O';
            op_cands[oi][n_cands[oi]++] = 'G';
            op_cands[oi][n_cands[oi]++] = 'C';
            op_cands[oi][n_cands[oi]++] = 'R';
        } else if (is_concat[op]) {
            op_cands[oi][n_cands[oi]++] = 'C';
        } else if (is_revconcat[op]) {
            op_cands[oi][n_cands[oi]++] = 'R';
        } else {
            // Filter by observed output lengths for this operator
            int mx = max_olen[op], mn = min_olen[op];
            int neg = has_neg[op];
            // Adjust: if neg, actual digit count is olen-1
            int effective_mn = neg ? (mn > 1 ? mn - 1 : mn) : mn;

            if (mx >= 4) {
                op_cands[oi][n_cands[oi]++] = '*';
            }
            if (mx >= 3) {
                op_cands[oi][n_cands[oi]++] = '+';
            }
            if (mx >= 3 && mn <= 3) {
                op_cands[oi][n_cands[oi]++] = '*'; // mul can give 3 digits too
            }
            if (effective_mn <= 2 || neg) {
                op_cands[oi][n_cands[oi]++] = '-';
                op_cands[oi][n_cands[oi]++] = 'r';
                op_cands[oi][n_cands[oi]++] = 'A';
                op_cands[oi][n_cands[oi]++] = 'M';
            }
            if (effective_mn <= 1) {
                op_cands[oi][n_cands[oi]++] = '%';
                op_cands[oi][n_cands[oi]++] = 'D';
                op_cands[oi][n_cands[oi]++] = '/';
                op_cands[oi][n_cands[oi]++] = 'G';
            }
            op_cands[oi][n_cands[oi]++] = 'X';
            op_cands[oi][n_cands[oi]++] = '&';
            op_cands[oi][n_cands[oi]++] = 'O';
            // Deduplicate
            int deduped = 0;
            for (int j = 0; j < n_cands[oi]; j++) {
                int dup = 0;
                for (int k = 0; k < deduped; k++)
                    if (op_cands[oi][k] == op_cands[oi][j]) { dup = 1; break; }
                if (!dup) op_cands[oi][deduped++] = op_cands[oi][j];
            }
            n_cands[oi] = deduped;
        }
    }

    // Generate combinations
    if (n_uops == 1) {
        for (int a = 0; a < n_cands[0] && *n_combos < MAX_COMBOS; a++) {
            op_combos[*n_combos][0] = op_cands[0][a]; (*n_combos)++;
        }
    } else if (n_uops == 2) {
        for (int a = 0; a < n_cands[0]; a++)
            for (int b = 0; b < n_cands[1] && *n_combos < MAX_COMBOS; b++) {
                op_combos[*n_combos][0] = op_cands[0][a];
                op_combos[*n_combos][1] = op_cands[1][b]; (*n_combos)++;
            }
    } else if (n_uops == 3) {
        for (int a = 0; a < n_cands[0]; a++)
            for (int b = 0; b < n_cands[1]; b++)
                for (int c = 0; c < n_cands[2] && *n_combos < MAX_COMBOS; c++) {
                    op_combos[*n_combos][0] = op_cands[0][a];
                    op_combos[*n_combos][1] = op_cands[1][b];
                    op_combos[*n_combos][2] = op_cands[2][c]; (*n_combos)++;
                }
    } else if (n_uops == 4) {
        for (int a = 0; a < n_cands[0]; a++)
            for (int b = 0; b < n_cands[1]; b++)
                for (int c = 0; c < n_cands[2]; c++)
                    for (int d = 0; d < n_cands[3] && *n_combos < MAX_COMBOS; d++) {
                        op_combos[*n_combos][0] = op_cands[0][a];
                        op_combos[*n_combos][1] = op_cands[1][b];
                        op_combos[*n_combos][2] = op_cands[2][c];
                        op_combos[*n_combos][3] = op_cands[3][d]; (*n_combos)++;
                    }
    }
}

int solve() {
    int unique_ops[8], n_uops = 0;
    for (int i = 0; i < n_ex; i++) {
        int op = ex_op[i], found = 0;
        for (int j = 0; j < n_uops; j++) if (unique_ops[j] == op) { found = 1; break; }
        if (!found) unique_ops[n_uops++] = op;
    }
    { int found = 0;
      for (int j = 0; j < n_uops; j++) if (unique_ops[j] == q_op) { found = 1; break; }
      if (!found) unique_ops[n_uops++] = q_op;
    }
    if (n_uops > 4) return 0;

    static int op_combos[MAX_COMBOS][5];
    int n_combos;
    gen_combos(unique_ops, n_uops, op_combos, &n_combos);

    // Smart base selection
    int bases[16], n_bases = 0;
    if (n_sym >= 8) { bases[n_bases++] = n_sym; if (n_sym+1 <= 16) bases[n_bases++] = n_sym+1; }
    for (int b = 10; b <= 16; b += 6) {
        if (n_sym <= b) { int dup=0; for(int i=0;i<n_bases;i++) if(bases[i]==b) dup=1; if(!dup) bases[n_bases++]=b; }
    }
    for (int b = n_sym; b <= n_sym+3 && b <= 16; b++) {
        int dup=0; for(int i=0;i<n_bases;i++) if(bases[i]==b) dup=1; if(!dup) bases[n_bases++]=b;
    }

    for (int bi = 0; bi < n_bases; bi++) {
        base_g = bases[bi];
        if (n_sym > base_g) continue;

        for (int le = 0; le <= 1; le++) {
            is_le = le;

            for (int ci = 0; ci < n_combos; ci++) {
                if (timed_out) return 0;

                memset(opassign, 0, sizeof(opassign));
                for (int oi = 0; oi < n_uops; oi++)
                    opassign[unique_ops[oi]] = op_combos[ci][oi];

                n_arith = 0;
                for (int i = 0; i < n_ex; i++) {
                    int ot = opassign[ex_op[i]];
                    if (ot != 'C' && ot != 'R')
                        arith_idx[n_arith++] = i;
                }

                // Quick concat check
                int concat_ok = 1;
                for (int i = 0; i < n_ex; i++) {
                    int ot = opassign[ex_op[i]];
                    if (ot == 'C') {
                        char p[5]; p[0]=ex_l0[i]; p[1]=ex_l1[i]; p[2]=ex_r0[i]; p[3]=ex_r1[i]; p[4]=0;
                        if (ex_olen[i] != 4 || memcmp(p, ex_out[i], 4) != 0) { concat_ok = 0; break; }
                    } else if (ot == 'R') {
                        char p[5]; p[0]=ex_r0[i]; p[1]=ex_r1[i]; p[2]=ex_l0[i]; p[3]=ex_l1[i]; p[4]=0;
                        if (ex_olen[i] != 4 || memcmp(p, ex_out[i], 4) != 0) { concat_ok = 0; break; }
                    }
                }
                if (!concat_ok) continue;

                int qot = opassign[q_op];
                if (qot == 'C') {
                    char p[5]; p[0]=q_l0; p[1]=q_l1; p[2]=q_r0; p[3]=q_r1; p[4]=0;
                    if (ans_len == 4 && memcmp(p, answer, 4) == 0) return 1;
                    continue;
                }
                if (qot == 'R') {
                    char p[5]; p[0]=q_r0; p[1]=q_r1; p[2]=q_l0; p[3]=q_l1; p[4]=0;
                    if (ans_len == 4 && memcmp(p, answer, 4) == 0) return 1;
                    continue;
                }

                // Compute optimal symbol ordering for this config
                compute_sym_order();

                init_state();
                if (dfs(0)) return 1;
            }
        }
    }
    return 0;
}

/* ===== DERIVE_TRACE: emit the winning-path map-derivation trace (HC1) ===== */
long trace_budget;
int trace_complete(int idx) {  // does current partial m[] extend to a full assignment satisfying all arith examples?
    if (trace_budget-- <= 0) return 1;             // budget out -> assume viable (never over-eliminate)
    if (idx == n_sym) { for (int i=0;i<n_arith;i++) if(!check_partial_enhanced(arith_idx[i])) return 0; return 1; }
    int s = sym_order[idx];
    if (m[s] >= 0) return trace_complete(idx+1);    // fixed prefix symbol
    for (int d=0; d<base_g; d++) {
        if (used[d]) continue;
        m[s]=d; inv[d]=s; used[d]=1;
        int ok=1; for (int i=0;i<n_arith && ok;i++) if(!check_partial_enhanced(arith_idx[i])) ok=0;
        if (ok && trace_complete(idx+1)) { m[s]=-1; inv[d]=-1; used[d]=0; return 1; }
        m[s]=-1; inv[d]=-1; used[d]=0;
    }
    return 0;
}
void emit_trace() {
    int savem[128], savei[32], saveu[32];
    memcpy(savem,m,sizeof(m)); memcpy(savei,inv,sizeof(inv)); memcpy(saveu,used,sizeof(used));
    for (int idx=0; idx<n_sym; idx++) {
        int s = sym_order[idx], wd = savem[s];
        for (int c=0;c<128;c++) m[c]=-1;
        for (int d=0;d<32;d++){ inv[d]=-1; used[d]=0; }
        for (int j=0;j<idx;j++){ int sj=sym_order[j]; m[sj]=savem[sj]; inv[savem[sj]]=sj; used[savem[sj]]=1; }
        printf("TRC %s %d %c %d", pid, idx, s, wd);
        for (int d=0; d<base_g; d++) {
            if (used[d] || d==wd) continue;
            m[s]=d; inv[d]=s; used[d]=1;
            int clash_ex=-1;
            for (int i=0;i<n_arith;i++){ int ei=arith_idx[i]; if (sym_in_eq[s][ei] && !check_partial_enhanced(ei)) {clash_ex=ei;break;} }
            if (clash_ex>=0) printf(" C%d@%d", d, clash_ex);
            else { trace_budget=2000000; if (!trace_complete(idx+1)) printf(" N%d", d); else printf(" V%d", d); }
            m[s]=-1; inv[d]=-1; used[d]=0;
        }
        printf("\n");
    }
    memcpy(m,savem,sizeof(m)); memcpy(inv,savei,sizeof(inv)); memcpy(used,saveu,sizeof(used));
    fflush(stdout);
}

int main() {
    char line[256];
    int total = 0, solved = 0;
    examples_only = getenv("EXAMPLES_ONLY") ? 1 : 0;  // hypothesis-test mode
    int derive_trace = getenv("DERIVE_TRACE") ? 1 : 0;

    while (fgets(line, sizeof(line), stdin)) {
        if (line[0] == '\n' || line[0] == '#') continue;
        line[strcspn(line, "\n")] = 0;
        strncpy(pid, line, 63);

        if (!fgets(line, sizeof(line), stdin)) break;
        n_ex = atoi(line);
        if (n_ex <= 0 || n_ex > MAXEX) continue;

        for (int i = 0; i < n_ex; i++) {
            if (!fgets(line, sizeof(line), stdin)) break;
            line[strcspn(line, "\n")] = 0;
            char *eq = strstr(line, " = ");
            if (!eq) continue;
            *eq = 0;
            char *inp = line; while (*inp == ' ') inp++;
            char *out = eq + 3;
            ex_l0[i] = (unsigned char)inp[0];
            ex_l1[i] = (unsigned char)inp[1];
            ex_op[i] = (unsigned char)inp[2];
            ex_r0[i] = (unsigned char)inp[3];
            ex_r1[i] = (unsigned char)inp[4];
            strncpy(ex_out[i], out, 7);
            ex_olen[i] = strlen(out);
        }

        if (!fgets(line, sizeof(line), stdin)) break;
        line[strcspn(line, "\n")] = 0;
        q_l0 = (unsigned char)line[0]; q_l1 = (unsigned char)line[1];
        q_op = (unsigned char)line[2];
        q_op_sym = q_op;
        q_r0 = (unsigned char)line[3]; q_r1 = (unsigned char)line[4];

        if (!fgets(line, sizeof(line), stdin)) break;
        line[strcspn(line, "\n")] = 0;
        strncpy(answer, line, 7);
        ans_len = strlen(answer);

        // Collect digit symbols
        int ss[128]; memset(ss, 0, sizeof(ss));
        for (int i = 0; i < n_ex; i++) {
            ss[ex_l0[i]] = 1; ss[ex_l1[i]] = 1;
            ss[ex_r0[i]] = 1; ss[ex_r1[i]] = 1;
            for (int j = 0; j < ex_olen[i]; j++) ss[(unsigned char)ex_out[i][j]] = 1;
        }
        ss[q_l0] = 1; ss[q_l1] = 1; ss[q_r0] = 1; ss[q_r1] = 1;
        for (int j = 0; j < ans_len; j++) ss[(unsigned char)answer[j]] = 1;
        for (int i = 0; i < n_ex; i++) ss[ex_op[i]] = 0;
        ss[q_op] = 0;
        // Only remove '-' if it's an operator; keep as digit if it appears in output
        int dash_is_op = 0;
        for (int i = 0; i < n_ex; i++) if (ex_op[i] == '-') dash_is_op = 1;
        if (q_op == '-') dash_is_op = 1;
        if (dash_is_op) ss['-'] = 0;

        n_sym = 0;
        for (int c = 0; c < 128; c++) if (ss[c]) syms[n_sym++] = c;

        total++;
        puzzle_start = time(NULL);
        timed_out = 0;
        dfs_nodes = 0;
        // Dynamic timeout — generous for fleet runs
        if (n_sym <= 8) puzzle_timeout = 300;           // 5 min
        else if (n_sym == 9) puzzle_timeout = 900;      // 15 min
        else if (n_sym == 10) puzzle_timeout = 3600;    // 1 hour
        else puzzle_timeout = 7200;                     // 2 hours
        // overnight: no 90s cap — examples-only gets the full per-size timeout (300/900/3600/7200)
        // so large-base puzzles get a real chance + uniqueness re-pass is more reliable.

        // Try multiple output modes: normal, reversed, offset ±1/±2
        int offsets[] = {0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5};
        int n_offsets = 11;
        int found_solution = 0;

        for (int rev = 0; rev <= 1 && !found_solution; rev++) {
            reverse_output = rev;
            for (int oi = 0; oi < n_offsets && !found_solution; oi++) {
                output_offset = offsets[oi];
                if (timed_out) break;

                int result = solve();
                if (examples_only) {
                    if (result > 0) {
                        found_solution = 1; solved++;
                        char pred1[64]; strncpy(pred1, g_pred, sizeof(pred1)-1); pred1[63]=0;
                        char map1[256]; strncpy(map1, g_map, sizeof(map1)-1); map1[255]=0;
                        int b1 = base_g, le1 = is_le, off1 = output_offset, rev1 = reverse_output;
                        // uniqueness: is there a DIFFERENT query prediction fitting the examples?
                        examples_only = 2; strncpy(g_avoid, pred1, sizeof(g_avoid)-1); g_avoid[63]=0;
                        int r2 = solve();
                        int uniq = (r2 <= 0);
                        examples_only = 1;
                        if (strcmp(pred1, answer) != 0)
                            printf("DOUBT %s pred=%s official=%s b%d%s%s uniq=%d MAP:%s\n", pid, pred1, answer, b1, le1?"_LE":"",
                                   off1? (rev1?"_revoff":"_off"):(rev1?"_rev":""), uniq, map1);
                        else
                            printf("OKEX %s pred=%s b%d%s\n", pid, pred1, b1, le1?"_LE":"");
                        fflush(stdout);
                    }
                    if (timed_out) break;
                    continue;
                }
                if (result > 0) {
                    found_solution = 1;
                    solved++;
                    const char *mode = "";
                    if (rev && output_offset) {
                        static char mbuf[32];
                        snprintf(mbuf, sizeof(mbuf), "_rev_off%+d", output_offset);
                        mode = mbuf;
                    } else if (rev) mode = "_rev";
                    else if (output_offset) {
                        static char mbuf[16];
                        snprintf(mbuf, sizeof(mbuf), "_off%+d", output_offset);
                        mode = mbuf;
                    }
                    printf("OK %s b%d%s%s MAP:", pid, base_g, is_le ? "_LE" : "", mode);
                    for (int i = 0; i < n_sym; i++) {
                        printf("%c=%d", sym_order[i], m[sym_order[i]]);
                        if (i < n_sym-1) printf(",");
                    }
                    printf(" OPS:");
                    int printed = 0;
                    for (int c = 0; c < 128; c++) {
                        if (opassign[c] != 0) {
                            if (printed) printf(",");
                            char ot = opassign[c];
                            const char *oname = ot=='+' ? "add" : ot=='-' ? "sub" : ot=='r' ? "revsub" : ot=='*' ? "mul" : ot=='%' ? "mod" : ot=='M' ? "maxmod" : ot=='/' ? "div" : ot=='D' ? "maxdiv" : ot=='A' ? "absdiff" : ot=='X' ? "xor" : ot=='&' ? "and" : ot=='O' ? "or" : ot=='G' ? "gcd" : ot=='C' ? "concat" : ot=='R' ? "revconcat" : "?";
                            printf("%c=%s", c, oname);
                            printed++;
                        }
                    }
                    int qot = opassign[q_op];
                    if (qot == 'C' || qot == 'R') {
                        printf(" Q:concat");
                    } else {
                        int qlv = to_val(q_l0, q_l1);
                        int qrv = to_val(q_r0, q_r1);
                        int qval = compute(qlv, qrv, qot);
                        char opch = qot=='+' ? '+' : qot=='-' ? '-' : qot=='r' ? '-' : qot=='%' ? '%' : qot=='M' ? '%' : qot=='/' ? '/' : qot=='D' ? '/' : qot=='A' ? '|' : qot=='X' ? '^' : qot=='&' ? '&' : qot=='O' ? '|' : qot=='G' ? 'g' : '*';
                        printf(" Q:%d%c%d=%d", qlv, opch, qrv, qval);
                    }
                    printf("\n");
                    fflush(stdout);
                    if (derive_trace) emit_trace();
                }
                if (timed_out) break; // don't try more modes if timeout
            }
        }
        if (!found_solution && timed_out) {
            fprintf(stderr, "TIMEOUT %s (%d syms, %lld nodes)\n", pid, n_sym, dfs_nodes);
        }
        // Reset for next puzzle
        output_offset = 0;
        reverse_output = 0;

        if (total % 50 == 0)
            fprintf(stderr, "... %d/%d solved %d\n", total, total, solved);
    }

    fprintf(stderr, "Done: %d/%d solved\n", solved, total);
    return 0;
}
