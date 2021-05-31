package goblas

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/whipstein/golinalg/mat"
)

func TestDblasLevel3(t *testing.T) {
	var diag, diags mat.MatDiag
	var side, sides mat.MatSide
	var tranas, transa, transb, trans, transs mat.MatTrans
	var uplo, uplos mat.MatUplo
	var fatal, left, null, tran, same, upper bool
	var alpha, als, beta, bls, bets, err float64
	var i, j, n int
	var _a, __a, _ab, _c, _cc, a, aa, ab, b, bb, c, cc, w *mat.Matrix
	var ct, g *mat.Vector
	ok := true
	reset := true
	eps := epsilonf64()
	thresh := 16.0
	errmax := 0.0
	nmax := 65
	idim := []int{0, 1, 2, 3, 5, 9}
	alf := []float64{0.0, 1.0, 0.7}
	bet := []float64{0.0, 1.0, 1.3}
	isame := make([]bool, 13)
	ichd := []mat.MatDiag{mat.Unit, mat.NonUnit}
	ichs := []mat.MatSide{mat.Left, mat.Right}
	icht := []mat.MatTrans{mat.NoTrans, mat.Trans, mat.ConjTrans}
	ichu := []mat.MatUplo{mat.Upper, mat.Lower}
	optsfull := opts.DeepCopy()
	optsge := opts.DeepCopy()
	a = mf(nmax, 2*nmax, optsfull)
	ab = mf(nmax, 2*nmax, optsfull)
	b = mf(nmax, 2*nmax, optsfull)
	c = mf(nmax, nmax, optsfull)
	cc = mf(nmax, nmax, optsfull)
	ct = vf(nmax * nmax)
	g = vf(nmax)
	snames := []string{"Dgemm", "Dsymm", "Dtrmm", "Dtrsm", "Dsyrk", "Dsyr2k"}

	n = minint(32, nmax)
	for j = 1; j <= n; j++ {
		for i = 1; i <= n; i++ {
			a.Set(i-1, j-1, float64(maxint(i-j+1, 0)))
			ab.Set(i-1, j-1, float64(maxint(i-j+1, 0)))
			b.Set(i-1, j-1, float64(maxint(i-j+1, 0)))
		}
		a.Set(j-1, nmax, float64(j))
		a.Set(0, nmax+j-1, float64(j))
		ab.Set(j-1, nmax, float64(j))
		ab.Set(0, nmax+j-1, float64(j))
		b.Set(j-1, nmax, float64(j))
		b.Set(0, nmax+j-1, float64(j))
		c.Set(j-1, 0, 0.0)
	}
	for j = 1; j <= n; j++ {
		cc.Set(j-1, 0, float64(j*((j+1)*j))/2-float64((j+1)*j*(j-1))/3)
	}
	//     CC holds the exact result. On exit from SMMCH CT holds
	//     the result computed by SMMCH.
	_a = a.CopyIdx(nmax * nmax)
	dmmch(mat.NoTrans, mat.NoTrans, n, 1, n, 1.0, a, nmax, _a, nmax, 0.0, c, nmax, ct, g, cc, nmax, eps, &err, &fatal, true, t)
	if err != 0 {
		t.Errorf(" ERROR IN DMMCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n SMMCH WAS CALLED WITH TRANSA = %c AND TRANSB = %c\n AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", 'N', 'N', same, err)
	}
	dmmch(mat.NoTrans, mat.Trans, n, 1, n, 1.0, a, nmax, _a, nmax, 0.0, c, nmax, ct, g, cc, nmax, eps, &err, &fatal, true, t)
	if err != 0 {
		t.Errorf(" ERROR IN DMMCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n SMMCH WAS CALLED WITH TRANSA = %c AND TRANSB = %c\n AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", 'N', 'T', same, err)
	}
	for j = 1; j <= n; j++ {
		a.Set(j-1, nmax, float64(n-j+1))
		a.Set(0, nmax+j-1, float64(n-j+1))
		ab.Set(j-1, nmax, float64(n-j+1))
		ab.Set(0, nmax+j-1, float64(n-j+1))
		b.Set(j-1, nmax, float64(n-j+1))
		b.Set(0, nmax+j-1, float64(n-j+1))
	}
	for j = 1; j <= n; j++ {
		cc.Set(n-j+1-1, 0, float64(j*((j+1)*j))/2-float64((j+1)*j*(j-1))/3)
	}
	_a = a.CopyIdx(nmax * nmax)
	dmmch(mat.Trans, mat.NoTrans, n, 1, n, 1.0, a, nmax, _a, nmax, 0.0, c, nmax, ct, g, cc, nmax, eps, &err, &fatal, true, t)
	if err != 0 {
		t.Errorf(" ERROR IN DMMCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n SMMCH WAS CALLED WITH TRANSA = %c AND TRANSB = %c\n AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", 'T', 'N', same, err)
	}
	dmmch(mat.Trans, mat.Trans, n, 1, n, 1.0, a, nmax, _a, nmax, 0.0, c, nmax, ct, g, cc, nmax, eps, &err, &fatal, true, t)
	if err != 0 {
		t.Errorf(" ERROR IN DMMCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n SMMCH WAS CALLED WITH TRANSA = %c AND TRANSB = %c\n AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", 'T', 'T', same, err)
	}

	for _, sname := range snames {
		fatal = false
		reset = true
		ok = true
		errmax = 0.0
		if sname == "Dgemm" {
			var trana, tranb bool
			var i, k, ks, laa, lbb, lcc, lda, ldas, ldb, ldbs, ldc, ldcs, m, ma, mb, ms, n, na, nb, ns int
			_, _, _ = laa, lbb, lcc
			var nargs int = 13
			var nc int = 0

			opts.Style = mat.General
			optsfull.Style = mat.General

			for _, m = range idim {

				for _, n = range idim {
					//           Set LDC to 1 more than minimum value if room.
					ldc = m
					if ldc < nmax {
						ldc = ldc + 1
					}
					//           Skip tests if not enough room.
					if ldc > nmax {
						continue
					}
					lcc = ldc * n
					null = n <= 0 || m <= 0

					for _, k = range idim {

						for _, transa = range mat.TransIter() {
							trana = transa.IsTrans()

							if trana {
								ma = k
								na = m
							} else {
								ma = m
								na = k
							}
							//                 Set LDA to 1 more than minimum value if room.
							lda = ma
							if lda < nmax {
								lda = lda + 1
							}
							//                 Skip tests if not enough room.
							if lda > nmax {
								continue
							}
							laa = lda * na

							//                 Generate the matrix A.
							a = mf(nmax, nmax, optsfull)
							aa = mf(lda, na, opts)
							dmakeL3(mat.General, mat.Lower, mat.NonUnit, ma, na, a, nmax, aa, lda, &reset, 0.0)

							for _, transb = range mat.TransIter() {
								tranb = transb.IsTrans()

								if tranb {
									mb = n
									nb = k
								} else {
									mb = k
									nb = n
								}
								//                    Set LDB to 1 more than minimum value if room.
								ldb = mb
								if ldb < nmax {
									ldb = ldb + 1
								}
								//                    Skip tests if not enough room.
								if ldb > nmax {
									continue
								}
								lbb = ldb * nb

								//                    Generate the matrix B.
								b = mf(nmax, nmax, optsfull)
								bb = mf(ldb, nb, opts)
								dmakeL3(mat.General, mat.Lower, mat.NonUnit, mb, nb, b, nmax, bb, ldb, &reset, 0.0)

								for _, alpha = range alf {

									for _, beta = range bet {
										//                          Generate the matrix C.
										c = mf(nmax, nmax, optsfull)
										cc = mf(ldc, n, opts)
										dmakeL3(mat.General, mat.Lower, mat.NonUnit, m, n, c, nmax, cc, ldc, &reset, 0.0)

										nc++

										//                          Save every datum before calling the
										//                          subroutine.
										tranas := transa
										tranbs := transb
										ms = m
										ns = n
										ks = k
										als = alpha
										as := aa.DeepCopy()
										ldas = lda
										bs := bb.DeepCopy()
										ldbs = ldb
										bls = beta
										cs := cc.DeepCopy()
										ldcs = ldc

										//                          Call the subroutine.
										Dgemm(transa, transb, &m, &n, &k, &alpha, aa, &lda, bb, &ldb, &beta, cc, &ldc)

										//                          Check if error-exit was taken incorrectly.
										if !ok {
											t.Errorf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
											fatal = true
											goto label1120
										}

										//                          See what data changed inside subroutines.
										isame[0] = transa == tranas
										isame[1] = transb == tranbs
										isame[2] = ms == m
										isame[3] = ns == n
										isame[4] = ks == k
										isame[5] = als == alpha
										isame[6] = reflect.DeepEqual(aa, as)
										isame[7] = ldas == lda
										isame[8] = reflect.DeepEqual(bb, bs)
										isame[9] = ldbs == ldb
										isame[10] = bls == beta
										if null {
											isame[11] = reflect.DeepEqual(cc, cs)
										} else {
											isame[11] = lderesM(m, n, cs, cc, ldc)
										}
										isame[12] = ldcs == ldc

										//                          If data was incorrectly changed, report
										//                          and return.
										same = true
										for i = 1; i <= nargs; i++ {
											same = same && isame[i-1]
											if !isame[i-1] {
												t.Errorf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
											}
										}
										if !same {
											fatal = true
											goto label1120
										}

										if !null {
											//                             Check the result.
											dmmch(transa, transb, m, n, k, alpha, a, nmax, b, nmax, beta, c, nmax, ct, g, cc, ldc, eps, &err, &fatal, true, t)
											errmax = maxf64(errmax, err)
											//                             If got really bad answer, report and
											//                             return.
											if fatal {
												goto label1120
											}
										}

									}

								}

							}

						}

					}

				}

			}

			//     Report result.
			if errmax < thresh {
				passL2(sname, nc, 17496, t)
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label1120:
			;
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%v,%3d,%3d,%3d,%4.1f, A,%3d, B,%3d,%4.1f, C,%3d).\n", sname, nc, sname, transa.String(), transb.String(), m, n, k, alpha, lda, ldb, beta, ldc)
		} else if sname == "Dsymm" {
			var i, lda, ldas, ldb, ldbs, ldc, ldcs, m, ms, n, na, ns int
			var nargs int = 12
			var nc int = 0

			opts.Style = mat.Symmetric
			optsfull.Style = mat.Symmetric

			for _, m = range idim {

				for _, n = range idim {
					//           Set LDC to 1 more than minimum value if room.
					ldc = m
					if ldc < nmax {
						ldc = ldc + 1
					}
					//           Skip tests if not enough room.
					if ldc > nmax {
						continue
					}
					// lcc = ldc * n
					null = n <= 0 || m <= 0

					//           Set LDB to 1 more than minimum value if room.
					ldb = m
					if ldb < nmax {
						ldb = ldb + 1
					}
					//           Skip tests if not enough room.
					if ldb > nmax {
						continue
					}
					// lbb = ldb * n

					//           Generate the matrix B.
					b = mf(nmax, nmax, optsge)
					bb = mf(ldb, n, optsge)
					dmakeL3(mat.General, mat.Lower, mat.NonUnit, m, n, b, nmax, bb, ldb, &reset, 0.0)

					for _, side = range ichs {
						left = side == mat.Left

						if left {
							na = m
						} else {
							na = n
						}
						//              Set LDA to 1 more than minimum value if room.
						lda = na
						if lda < nmax {
							lda = lda + 1
						}
						//              Skip tests if not enough room.
						if lda > nmax {
							goto label280
						}
						// laa = lda * na

						for _, uplo = range ichu {
							opts.Uplo = uplo
							optsfull.Uplo = uplo

							//                 Generate the symmetric matrix A.
							a = mf(nmax, nmax, optsfull)
							aa = mf(lda, na, opts)
							dmakeL3(mat.Symmetric, uplo, mat.NonUnit, na, na, a, nmax, aa, lda, &reset, 0.0)

							for _, alpha = range alf {

								for _, beta = range bet {

									//                       Generate the matrix C.
									c = mf(nmax, nmax, optsge)
									cc = mf(ldc, n, optsge)
									dmakeL3(mat.General, mat.Lower, mat.NonUnit, m, n, c, nmax, cc, ldc, &reset, 0.0)

									nc++
									//                       Save every datum before calling the
									//                       subroutine.
									sides = side
									uplos = uplo
									ms = m
									ns = n
									als = alpha
									as := aa.DeepCopy()
									ldas = lda
									bs := bb.DeepCopy()
									ldbs = ldb
									bls = beta
									cs := cc.DeepCopy()
									ldcs = ldc

									//                       Call the subroutine.
									Dsymm(side, uplo, &m, &n, &alpha, aa, &lda, bb, &ldb, &beta, cc, &ldc)

									//                       Check if error-exit was taken incorrectly.
									if !ok {
										t.Errorf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
										fatal = true
										goto label2110
									}
									//                       See what data changed inside subroutines.
									isame[0] = sides == side
									isame[1] = uplos == uplo
									isame[2] = ms == m
									isame[3] = ns == n
									isame[4] = als == alpha
									isame[5] = reflect.DeepEqual(aa, as)
									isame[6] = ldas == lda
									isame[7] = reflect.DeepEqual(bb, bs)
									isame[8] = ldbs == ldb
									isame[9] = bls == beta
									if null {
										isame[10] = reflect.DeepEqual(cc, cs)
									} else {
										isame[10] = lderesM(m, n, cs, cc, ldc)
									}
									isame[11] = ldcs == ldc
									//                       If data was incorrectly changed, report and
									//                       return.
									same = true
									for i = 1; i <= nargs; i++ {
										same = same && isame[i-1]
										if !isame[i-1] {
											t.Errorf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
										}
									}
									if !same {
										fatal = true
										goto label2110
									}

									if !null {
										//                          Check the result.
										if left {
											dmmch(mat.NoTrans, mat.NoTrans, m, n, m, alpha, a, nmax, b, nmax, beta, c, nmax, ct, g, cc, ldc, eps, &err, &fatal, true, t)
										} else {
											dmmch(mat.NoTrans, mat.NoTrans, m, n, n, alpha, b, nmax, a, nmax, beta, c, nmax, ct, g, cc, ldc, eps, &err, &fatal, true, t)
										}
										errmax = maxf64(errmax, err)
										//                          If got really bad answer, report and
										//                          return.
										if fatal {
											goto label2110
										}
									}

								}

							}

						}

					label280:
					}

				}
			}

			//     Report result.
			if errmax < thresh {
				passL2(sname, nc, 1296, t)
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label2110:
			;
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%v,%3d,%3d,%4.1f, A,%3d, B,%3d,%4.1f, C,%3d)    .\n", sname, nc, sname, side.String(), uplo.String(), m, n, alpha, lda, ldb, beta, ldc)
		} else if sname == "Dtrmm" || sname == "Dtrsm" {
			var i, j, laa, lbb, lda, ldas, ldb, ldbs, m, ms, n, na, ns int
			_, _ = laa, lbb
			var nargs int = 11
			var nc int = 0

			opts.Style = mat.Triangular
			optsfull.Style = mat.Triangular

			//     Set up zero matrix for SMMCH.
			for j = 1; j <= nmax; j++ {
				for i = 1; i <= nmax; i++ {
					c.Set(i-1, j-1, 0)
				}
			}

			for _, m = range idim {

				for _, n = range idim {
					//           Set LDB to 1 more than minimum value if room.
					ldb = m
					if ldb < nmax {
						ldb = ldb + 1
					}
					//           Skip tests if not enough room.
					if ldb > nmax {
						goto label3130
					}
					lbb = ldb * n
					null = m <= 0 || n <= 0

					for _, side = range ichs {
						opts.Side = side
						optsfull.Side = side
						left = side == mat.Left
						if left {
							na = m
						} else {
							na = n
						}
						//              Set LDA to 1 more than minimum value if room.
						lda = na
						if lda < nmax {
							lda++
						}
						//              Skip tests if not enough room.
						if lda > nmax {
							goto label3130
						}
						laa = lda * na

						for _, uplo = range ichu {
							opts.Uplo = uplo
							optsfull.Uplo = uplo

							for _, transa = range icht {

								for _, diag = range ichd {
									opts.Diag = diag
									optsfull.Diag = diag

									for _, alpha = range alf {

										//                          Generate the matrix A.
										a = mf(nmax, nmax, optsfull)
										aa = mf(lda, na, opts)
										dmakeL3(mat.Triangular, uplo, diag, na, na, a, nmax, aa, lda, &reset, 0.0)

										//                          Generate the matrix B.
										b = mf(nmax, nmax, optsge)
										bb = mf(ldb, n, optsge)
										dmakeL3(mat.General, mat.Lower, mat.NonUnit, m, n, b, nmax, bb, ldb, &reset, 0.0)

										nc++
										//                          Save every datum before calling the
										//                          subroutine.
										sides = side
										uplos = uplo
										tranas = transa
										diags = diag
										ms = m
										ns = n
										als = alpha
										as := aa.DeepCopy()
										ldas = lda
										bs := bb.DeepCopy()
										ldbs = ldb

										//                          Call the subroutine.
										if sname[3:5] == "mm" {
											Dtrmm(side, uplo, transa, diag, &m, &n, &alpha, aa, &lda, bb, &ldb)
										} else if sname[3:5] == "sm" {
											Dtrsm(side, uplo, transa, diag, &m, &n, &alpha, aa, &lda, bb, &ldb)
										}

										//                          Check if error-exit was taken incorrectly.
										if !ok {
											t.Errorf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
											fatal = true
											goto label3150
										}

										//                          See what data changed inside subroutines.
										isame[0] = sides == side
										isame[1] = uplos == uplo
										isame[2] = tranas == transa
										isame[3] = diags == diag
										isame[4] = ms == m
										isame[5] = ns == n
										isame[6] = als == alpha
										isame[7] = reflect.DeepEqual(aa, as)
										isame[8] = ldas == lda
										if null {
											isame[9] = reflect.DeepEqual(bb, bs)
										} else {
											isame[9] = lderesM(m, n, bs, bb, ldb)
										}
										isame[10] = ldbs == ldb
										//
										//                          If data was incorrectly changed, report and
										//                          return.
										//
										same = true
										for i = 1; i <= nargs; i++ {
											same = same && isame[i-1]
											if !isame[i-1] {
												t.Errorf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
											}
										}
										if !same {
											fatal = true
											goto label3150
										}
										//
										if !null {
											if sname[3:5] == "mm" {
												//
												//                                Check the result.
												//
												if left {
													dmmch(transa, mat.NoTrans, m, n, m, alpha, a, nmax, b, nmax, 0.0, c, nmax, ct, g, bb, ldb, eps, &err, &fatal, true, t)
												} else {
													dmmch(mat.NoTrans, transa, m, n, n, alpha, b, nmax, a, nmax, 0.0, c, nmax, ct, g, bb, ldb, eps, &err, &fatal, true, t)
												}
											} else if sname[3:5] == "sm" {
												//                                Compute approximation to original
												//                                matrix.
												for j := 1; j <= n; j++ {
													for i := 1; i <= m; i++ {
														c.Set(i-1, j-1, bb.Get(i-1, j-1))
														bb.Set(i-1, j-1, alpha*b.Get(i-1, j-1))
													}
												}

												if left {
													dmmch(transa, mat.NoTrans, m, n, m, 1.0, a, nmax, c, nmax, 0.0, b, nmax, ct, g, bb, ldb, eps, &err, &fatal, false, t)
												} else {
													dmmch(mat.NoTrans, transa, m, n, n, 1.0, c, nmax, a, nmax, 0.0, b, nmax, ct, g, bb, ldb, eps, &err, &fatal, false, t)
												}
											}
											errmax = maxf64(errmax, err)
											//                             If got really bad answer, report and
											//                             return.
											if fatal {
												goto label3150
											}
										}

									}

								}

							}

						}

					}

				label3130:
				}

			}

			//     Report result.
			if errmax < thresh {
				passL2(sname, nc, 2592, t)
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label3150:
			;
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%v,%v,%v,%3d,%3d,%4.1f, A,%3d, B,%3d)        .\n", sname, nc, sname, side.String(), uplo.String(), transa.String(), diag.String(), m, n, alpha, lda, ldb)
		} else if sname == "Dsyrk" {
			var i, j, jc, jj, k, ks, laa, lcc, lda, ldas, ldc, ldcs, lj, ma, n, na, ns int
			_, _ = laa, lcc
			var nargs int = 10
			var nc int = 0

			opts.Style = mat.Symmetric
			optsfull.Style = mat.Symmetric

			for _, n = range idim {
				//        Set LDC to 1 more than minimum value if room.
				ldc = n
				if ldc < nmax {
					ldc++
				}
				//        Skip tests if not enough room.
				if ldc > nmax {
					continue
				}
				lcc = ldc * n
				null = n <= 0

				for _, k = range idim {

					for _, trans = range icht {
						tran = trans.IsTrans()
						if tran {
							ma = k
							na = n
						} else {
							ma = n
							na = k
						}
						//              Set LDA to 1 more than minimum value if room.
						lda = ma
						if lda < nmax {
							lda++
						}
						//              Skip tests if not enough room.
						if lda > nmax {
							continue
						}
						laa = lda * na

						//              Generate the matrix A.
						a = mf(nmax, nmax, optsge)
						aa = mf(lda, na, optsge)
						dmakeL3(mat.General, mat.Lower, mat.NonUnit, ma, na, a, nmax, aa, lda, &reset, 0.0)

						for _, uplo = range ichu {
							opts.Uplo = uplo
							optsfull.Uplo = uplo
							upper = uplo == mat.Upper

							for _, alpha = range alf {

								for _, beta = range bet {
									//                       Generate the matrix C.
									c = mf(nmax, nmax, optsfull)
									cc = mf(ldc, n, opts)
									dmakeL3(mat.Symmetric, uplo, mat.NonUnit, n, n, c, nmax, cc, ldc, &reset, 0.0)

									//                       Increment test counter
									nc++

									//                       Save every datum before calling the subroutine.
									uplos = uplo
									transs = trans
									ns = n
									ks = k
									als = alpha
									as := aa.DeepCopy()
									ldas = lda
									bets = beta
									cs := cc.DeepCopy()
									ldcs = ldc

									//                       Call the subroutine.
									Dsyrk(uplo, trans, &n, &k, &alpha, aa, &lda, &beta, cc, &ldc)

									//                       Check if error-exit was taken incorrectly.
									if !ok {
										t.Errorf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
										fatal = true
										goto label4120
									}
									//                       See what data changed inside subroutines.
									isame[0] = uplos == uplo
									isame[1] = transs == trans
									isame[2] = ns == n
									isame[3] = ks == k
									isame[4] = als == alpha
									isame[5] = reflect.DeepEqual(aa, as)
									isame[6] = ldas == lda
									isame[7] = bets == beta
									if null {
										isame[8] = reflect.DeepEqual(cc, cs)
									} else {
										isame[8] = lderesM(n, n, cs, cc, ldc)
									}
									isame[9] = ldcs == ldc
									//                       If data was incorrectly changed, report and
									//                       return.
									same = true
									for i = 1; i <= nargs; i++ {
										same = same && isame[i-1]
										if !isame[i-1] {
											t.Errorf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
										}
									}
									if !same {
										fatal = true
										goto label4120
									}

									if !null {
										//                          Check the result column by column.
										jc = 1
										for j = 1; j <= n; j++ {
											if upper {
												jj = 1
												lj = j
											} else {
												jj = j
												lj = n - j + 1
											}
											_c = c.CopyIdx(jj - 1 + (j-1)*nmax)
											_cc = cc.CopyIdx(jc - 1)
											if tran {
												_a = a.CopyIdx(0 + (jj-1)*nmax)
												__a = a.CopyIdx(0 + (j-1)*nmax)
												dmmch(mat.Trans, mat.NoTrans, lj, 1, k, alpha, _a, nmax, __a, nmax, beta, _c, nmax, ct, g, _cc, ldc, eps, &err, &fatal, true, t)
											} else {
												_a = a.CopyIdx(jj - 1 + (0)*nmax)
												__a = a.CopyIdx(j - 1 + (0)*nmax)
												dmmch(mat.NoTrans, mat.Trans, lj, 1, k, alpha, _a, nmax, __a, nmax, beta, _c, nmax, ct, g, _cc, ldc, eps, &err, &fatal, true, t)
											}
											if upper {
												jc += ldc
											} else {
												jc += ldc + 1
											}
											errmax = maxf64(errmax, err)
											//                             If got really bad answer, report and
											//                             return.
											if fatal {
												goto label4110
											}
										}
									}

								}

							}

						}

					}

				}

			}

			//     Report result.
			if errmax < thresh {
				passL2(sname, nc, 1944, t)
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label4110:
			;
			if n > 1 {
				fmt.Printf("      THESE ARE THE RESULTS FOR COLUMN %3d\n", j)
			}

		label4120:
			;
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%v,%3d,%3d,%4.1f, A,%3d,%4.1f, C,%3d)           .\n", sname, nc, sname, uplo.String(), trans.String(), n, k, alpha, lda, beta, ldc)
		} else if sname == "Dsyr2k" {
			var i, j, jc, jj, jjab, k, ks, laa, lbb, lcc, lda, ldas, ldb, ldbs, ldc, ldcs, lj, ma, n, na, ns int
			_, _, _ = laa, lbb, lcc
			var nargs int = 12
			var nc int = 0

			opts.Style = mat.Symmetric
			optsfull.Style = mat.Symmetric

			for _, n = range idim {
				//        Set LDC to 1 more than minimum value if room.
				ldc = n
				if ldc < nmax {
					ldc++
				}
				//        Skip tests if not enough room.
				if ldc > nmax {
					goto label5130
				}
				lcc = ldc * n
				null = n <= 0

				for _, k = range idim {

					for _, trans = range icht {
						tran = trans.IsTrans()
						if tran {
							ma = k
							na = n
						} else {
							ma = n
							na = k
						}
						//              Set LDA to 1 more than minimum value if room.
						lda = ma
						if lda < nmax {
							lda++
						}
						//              Skip tests if not enough room.
						if lda > nmax {
							continue
						}
						laa = lda * na

						//              Generate the matrix A.
						aa = mf(lda, na, optsge)
						if tran {
							ab = mf(2*nmax, nmax, optsge)
							dmakeL3(mat.General, mat.Lower, mat.NonUnit, ma, na, ab, 2*nmax, aa, lda, &reset, 0.0)
						} else {
							ab = mf(nmax, nmax, optsge)
							dmakeL3(mat.General, mat.Lower, mat.NonUnit, ma, na, ab, nmax, aa, lda, &reset, 0.0)
						}

						//              Generate the matrix B.
						ldb = lda
						lbb = laa
						bb = mf(ldb, na, optsge)
						if tran {
							_ab = ab.CopyIdx(k + 1 - 1)
							dmakeL3(mat.General, mat.Lower, mat.NonUnit, ma, na, _ab, 2*nmax, bb, ldb, &reset, 0.0)
						} else {
							_ab = ab.CopyIdx(1 - 1 + (k)*nmax)
							dmakeL3(mat.General, mat.Lower, mat.NonUnit, ma, na, _ab, nmax, bb, ldb, &reset, 0.0)
						}

						for _, uplo = range ichu {
							opts.Uplo = uplo
							optsfull.Uplo = uplo
							upper = uplo == mat.Upper

							for _, alpha = range alf {

								for _, beta = range bet {
									//                       Generate the matrix C.
									c = mf(nmax, n, optsfull)
									cc = mf(ldc, n, optsfull)
									dmakeL3(mat.Symmetric, uplo, mat.NonUnit, n, n, c, nmax, cc, ldc, &reset, 0.0)

									//                       Increment test counter
									nc++

									//                       Save every datum before calling the subroutine.
									uplos = uplo
									transs = trans
									ns = n
									ks = k
									als = alpha
									as := aa.DeepCopy()
									ldas = lda
									bs := bb.DeepCopy()
									ldbs = ldb
									bets = beta
									cs := cc.DeepCopy()
									ldcs = ldc

									//                       Call the subroutine.
									Dsyr2k(uplo, trans, &n, &k, &alpha, aa, &lda, bb, &ldb, &beta, cc, &ldc)

									//                       Check if error-exit was taken incorrectly.
									if !ok {
										t.Errorf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
										fatal = true
										goto label5150
									}
									//                       See what data changed inside subroutines.
									isame[0] = uplos == uplo
									isame[1] = transs == trans
									isame[2] = ns == n
									isame[3] = ks == k
									isame[4] = als == alpha
									isame[5] = reflect.DeepEqual(aa, as)
									isame[6] = ldas == lda
									isame[7] = reflect.DeepEqual(bb, bs)
									isame[8] = ldbs == ldb
									isame[9] = bets == beta
									if null {
										isame[10] = reflect.DeepEqual(cc, cs)
									} else {
										isame[10] = lderesM(n, n, cs, cc, ldc)
									}
									isame[11] = ldcs == ldc
									//                       If data was incorrectly changed, report and
									//                       return.
									same = true
									for i = 1; i <= nargs; i++ {
										same = same && isame[i-1]
										if !isame[i-1] {
											t.Errorf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
										}
									}
									if !same {
										fatal = true
										goto label5150
									}

									if !null {
										//                          Check the result column by column.
										jjab = 1
										jc = 1
										w = mf(2*nmax, 1, optsge)
										for j = 1; j <= n; j++ {
											if upper {
												jj = 1
												lj = j
											} else {
												jj = j
												lj = n - j + 1
											}
											_c = c.CopyIdx(jj - 1 + (j-1)*nmax)
											_cc = cc.CopyIdx(jc - 1)
											if tran {
												for i = 1; i <= k; i++ {
													w.Set(i-1, 0, ab.Get(k+i-1, j-1))
													w.Set(k+i-1, 0, ab.Get(i-1, j-1))
												}
												_ab = ab.CopyIdx(jjab - 1)
												dmmch(mat.Trans, mat.NoTrans, lj, 1, 2*k, alpha, _ab, 2*nmax, w, 2*nmax, beta, _c, nmax, ct, g, _cc, ldc, eps, &err, &fatal, true, t)
											} else {
												for i = 1; i <= k; i++ {
													w.Set(i-1, 0, ab.Get(j-1, k+i-1))
													w.Set(k+i-1, 0, ab.Get(j-1, i-1))
												}
												_ab = ab.CopyIdx(jj - 1)
												dmmch(mat.NoTrans, mat.NoTrans, lj, 1, 2*k, alpha, _ab, nmax, w, 2*nmax, beta, _c, nmax, ct, g, _cc, ldc, eps, &err, &fatal, true, t)
											}
											if upper {
												jc += ldc
											} else {
												jc += ldc + 1
												if tran {
													jjab += 2 * nmax
												}
											}
											errmax = maxf64(errmax, err)
											//                             If got really bad answer, report and
											//                             return.
											if fatal {
												goto label5140
											}
										}
									}

								}

							}

						}

					}

				}

			label5130:
			}

			//     Report result.
			if errmax < thresh {
				passL2(sname, nc, 1944, t)
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label5140:
			;
			if n > 1 {
				fmt.Printf("      THESE ARE THE RESULTS FOR COLUMN %3d\n", j)
			}

		label5150:
			;
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%v,%3d,%3d,%4.1f, A,%3d, B,%3d,%4.1f, C,%3d)    .\n", sname, nc, sname, uplo.String(), trans.String(), n, k, alpha, lda, ldb, beta, ldc)
		}
	}
}

func dmmch(transa, transb mat.MatTrans, m, n, kk int, alpha float64, a *mat.Matrix, lda int, b *mat.Matrix, ldb int, beta float64, c *mat.Matrix, ldc int, ct, g *mat.Vector, cc *mat.Matrix, ldcc int, eps float64, err *float64, fatal *bool, mv bool, t *testing.T) {
	var trana, tranb bool
	var erri, one, zero float64
	var i, j, k int

	zero = 0.0
	one = 1.0

	trana = transa.IsTrans()
	tranb = transb.IsTrans()

	//     Compute expected result, one column at a time, in CT using data
	//     in A, B and C.
	//     Compute gauges in G.
	for j = 1; j <= n; j++ {

		for i = 1; i <= m; i++ {
			ct.Set(i-1, zero)
			g.Set(i-1, zero)
		}
		if !trana && !tranb {
			for k = 1; k <= kk; k++ {
				for i = 1; i <= m; i++ {
					ct.Set(i-1, ct.Get(i-1)+a.Get(i-1, k-1)*b.Get(k-1, j-1))
					g.Set(i-1, g.Get(i-1)+math.Abs(a.Get(i-1, k-1))*math.Abs(b.Get(k-1, j-1)))
				}
			}
		} else if trana && !tranb {
			for k = 1; k <= kk; k++ {
				for i = 1; i <= m; i++ {
					ct.Set(i-1, ct.Get(i-1)+a.Get(k-1, i-1)*b.Get(k-1, j-1))
					g.Set(i-1, g.Get(i-1)+math.Abs(a.Get(k-1, i-1))*math.Abs(b.Get(k-1, j-1)))
				}
			}
		} else if !trana && tranb {
			for k = 1; k <= kk; k++ {
				for i = 1; i <= m; i++ {
					ct.Set(i-1, ct.Get(i-1)+a.Get(i-1, k-1)*b.Get(j-1, k-1))
					g.Set(i-1, g.Get(i-1)+math.Abs(a.Get(i-1, k-1))*math.Abs(b.Get(j-1, k-1)))
				}
			}
		} else if trana && tranb {
			for k = 1; k <= kk; k++ {
				for i = 1; i <= m; i++ {
					ct.Set(i-1, ct.Get(i-1)+a.Get(k-1, i-1)*b.Get(j-1, k-1))
					g.Set(i-1, g.Get(i-1)+math.Abs(a.Get(k-1, i-1))*math.Abs(b.Get(j-1, k-1)))
				}
			}
		}
		for i = 1; i <= m; i++ {
			ct.Set(i-1, alpha*ct.Get(i-1)+beta*c.Get(i-1, j-1))
			g.Set(i-1, math.Abs(alpha)*g.Get(i-1)+math.Abs(beta)*math.Abs(c.Get(i-1, j-1)))
		}

		//        Compute the error ratio for this result.
		(*err) = zero
		for i = 1; i <= m; i++ {
			erri = math.Abs(ct.Get(i-1)-cc.Get(i-1, j-1)) / eps
			if g.Get(i-1) != zero {
				erri /= g.Get(i - 1)
			}
			(*err) = maxf64(*err, erri)
			if (*err)*math.Sqrt(eps) >= one {
				goto label130
			}
		}

	}

	//     If the loop completes, all results are at least half accurate.
	return

	//     Report fatal error.
label130:
	;
	*fatal = true
	t.Fail()
	fmt.Printf(" ******* FATAL ERROR - COMPUTED RESULT IS LESS THAN HALF ACCURATE *******\n           EXPECTED RESULT   COMPUTED RESULT\n")
	for i = 0; i < m; i++ {
		if mv {
			fmt.Printf(" %7d%18.6f%18.6f\n", i, ct.Get(i), cc.Get(i, j-1))
		} else {
			fmt.Printf(" %7d%18.6f%18.6f\n", i, cc.Get(i, j-1), ct.Get(i))
		}
	}
	if n > 1 {
		fmt.Printf("      THESE ARE THE RESULTS FOR COLUMN %3d\n", j-1)
	}

}

func dmakeL3(style mat.MatStyle, uplo mat.MatUplo, diag mat.MatDiag, m, n int, a *mat.Matrix, nmax int, aa *mat.Matrix, lda int, reset *bool, transl float64) {
	var gen, lower, sym, tri, unit, upper bool
	var one, rogue, zero float64
	var i, ibeg, iend, j int

	zero = 0.0
	one = 1.0
	rogue = -1.0e10

	gen = style == mat.General
	sym = style == mat.Symmetric
	tri = style == mat.Triangular
	upper = (sym || tri) && uplo == mat.Upper
	lower = (sym || tri) && uplo == mat.Lower
	unit = tri && diag == mat.Unit

	//     Generate data in array A.
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			if gen || (upper && i <= j) || (lower && i >= j) {
				a.Set(i-1, j-1, dbeg(reset)+transl)
				if i != j {
					//                 Set some elements to zero
					if n > 3 && j == n/2 {
						a.Set(i-1, j-1, zero)
					}
					if sym {
						a.Set(j-1, i-1, a.Get(i-1, j-1))
					} else if tri {
						a.Set(j-1, i-1, zero)
					}
				}
			}
		}
		if tri {
			a.Set(j-1, j-1, a.Get(j-1, j-1)+one)
		}
		if unit {
			a.Set(j-1, j-1, one)
		}
	}

	//     Store elements in array AS in data structure required by routine.
	if gen {
		for j = 1; j <= n; j++ {
			for i = 1; i <= m; i++ {
				aa.Set(i-1, j-1, a.Get(i-1, j-1))
			}
			for i = m + 1; i <= lda; i++ {
				aa.Set(i-1, j-1, rogue)
			}
		}
	} else if sym || tri {
		for j = 1; j <= n; j++ {
			if upper {
				ibeg = 1
				if unit {
					iend = j - 1
				} else {
					iend = j
				}
			} else {
				if unit {
					ibeg = j + 1
				} else {
					ibeg = j
				}
				iend = n
			}
			for i = 1; i <= ibeg-1; i++ {
				aa.Set(i-1, j-1, rogue)
			}
			for i = ibeg; i <= iend; i++ {
				aa.Set(i-1, j-1, a.Get(i-1, j-1))
			}
			for i = iend + 1; i <= lda; i++ {
				aa.Set(i-1, j-1, rogue)
			}
		}
	}
}
