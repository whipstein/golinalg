package goblas

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"testing"

	"github.com/whipstein/golinalg/mat"
)

func TestDblasLevel2(t *testing.T) {
	var fatal, null, reset, same, tran, upper bool
	var trans, transs mat.MatTrans
	var diag, diags mat.MatDiag
	var uplo, uplos mat.MatUplo
	var alpha, als, beta, bls, err, errmax float64
	var i, j, k, n int
	var ik, iku, im, incx, incy, kl, kls, ks, ku, kus, ja, jj, lda, ldas, lj, lx, ly, m, ml, ms, nargs, nc, nd, nk, nl, ns int
	var a, aa, as *mat.Matrix
	var g, x, xs, xt, xx, y, ys, yt, yy *mat.Vector
	ok := true
	idim := []int{0, 1, 2, 3, 5, 9}
	kb := []int{0, 1, 2, 4}
	inc := []int{1, 2, -1, -2}
	alf := []float64{0.0, 1.0, 0.7}
	bet := []float64{0.0, 1.0, 0.9}
	thresh := 16.0
	eps := epsilonf64()
	nmax := 65
	nkb := 4
	snames := []string{"Dgemv", "Dgbmv", "Dsymv", "Dsbmv", "Dspmv", "Dtrmv", "Dtbmv", "Dtpmv", "Dtrsv", "Dtbsv", "Dtpsv", "Dger", "Dsyr", "Dspr", "Dsyr2", "Dspr2"}
	isame := make([]bool, 13)
	ichd := []mat.MatDiag{mat.Unit, mat.NonUnit}
	icht := []mat.MatTrans{mat.NoTrans, mat.Trans, mat.ConjTrans}
	ichu := []mat.MatUplo{mat.Upper, mat.Lower}
	optsfull := opts.DeepCopy()
	a = mf(nmax, nmax, opts)
	g = vf(nmax)
	x = vf(nmax)
	xt = vf(nmax)
	y = vf(nmax)
	yt = vf(nmax)
	yy = vf(nmax)
	fmt.Printf("\n***** DBLAS Level 2 Tests *****\n")

	// n = min(32, nmax)
	n = nmax
	for j = 1; j <= n; j++ {
		for i = 1; i <= n; i++ {
			a.Set(i-1, j-1, float64(max(i-j+1, 0)))
		}
		x.Set(j-1, float64(j))
		y.Set(j-1, 0)
	}
	for j = 1; j <= n; j++ {
		yy.Set(j-1, float64(j*((j+1)*j))/2-float64(((j+1)*j*(j-1)))/3)
	}
	//     YY holds the exact result. On exit from SMVCH YT holds
	//     the result computed by SMVCH.
	dmvch(NoTrans, n, n, 1.0, a, nmax, x, 1, 0.0, y, 1, yt, g, yy, eps, &err, &fatal, true, t)
	if err != 0 {
		t.Errorf(" ERROR IN DMVCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n DMVCH WAS CALLED WITH TRANS = %c AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", trans, same, err)
	}
	dmvch(Trans, n, n, 1.0, a, nmax, x, -1, 0.0, y, -1, yt, g, yy, eps, &err, &fatal, true, t)
	if err != 0 {
		t.Errorf(" ERROR IN DMVCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n DMVCH WAS CALLED WITH TRANS = %c AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", trans, same, err)
	}
	for _, sname := range snames {
		fatal = false
		reset = true
		ok = true
		errmax = 0.0
		if sname == "Dgemv" || sname == "Dgbmv" {
			var full bool = sname[2] == 'e'
			var banded bool = sname[2] == 'b'

			for i := range isame {
				isame[i] = true
			}

			if dchkeLevel2(sname); !common.infoc.ok {
				t.Fail()
			}

			opts.Style = mat.General
			optsfull.Style = mat.General
			if full {
				nargs = 11
				opts.Storage = mat.Dense
			} else if banded {
				nargs = 13
				opts.Storage = mat.Banded
			}

			nc = 0

			for _, maj := range []mat.MatMajor{mat.Row, mat.Col} {
				opts.Major = maj
				optsfull.Major = maj

				for _, n = range idim {
					nd = n/2 + 1

					for im = 1; im <= 2; im++ {
						if im == 1 {
							m = max(n-nd, 0)
						}
						if im == 2 {
							m = min(n+nd, nmax)
						}

						if banded {
							nk = nkb
						} else {
							nk = 1
						}

						for iku = 1; iku <= nk; iku++ {
							if banded {
								ku = kb[iku-1]
								kl = max(ku-1, 0)
							} else {
								ku = n - 1
								kl = m - 1
							}
							opts.Kl, opts.Ku = kl, ku

							//              Set LDA to 1 more than minimum value if room.
							if banded {
								lda = kl + ku + 1
							} else {
								lda = m
							}
							if lda < nmax {
								lda++
							}
							//              Skip tests if not enough room.
							if lda > nmax {
								continue
							}
							null = n <= 0 || m <= 0

							//              Generate the matrix A.
							a = mf(nmax, n, optsfull)
							aa = mf(lda, n, opts)
							dmakeL2M(m, n, a, nmax, aa, lda, kl, ku, &reset, 0.0)

							for _, trans := range mat.IterMatTrans() {
								tran = trans.IsTrans()

								if tran {
									ml = n
									nl = m
								} else {
									ml = m
									nl = n
								}

								for _, incx = range inc {
									lx := abs(incx) * nl

									//                    Generate the vector X.
									x = vf(nl)
									xx = vf(lx)
									xiter := x.Iter(nl, sign(1, incx))
									xxiter := xx.Iter(nl, incx)
									dmakeL2V(1, nl, x, sign(1, incx), xx, incx, 0, nl-1, &reset, 0.5)
									if nl > 1 {
										x.Set(xiter[nl/2-1], 0)
										xx.Set(xxiter[nl/2-1], 0)
									}

									for _, incy = range inc {
										ly := abs(incy) * ml

										for _, alpha = range alf {

											for _, beta = range bet {
												//                             Generate the vector Y.
												y = vf(ml)
												yy = vf(ly)
												dmakeL2V(1, ml, y, sign(1, incy), yy, incy, 0, ml-1, &reset, 0.0)

												nc++

												//                             Save every datum before calling the
												//                             subroutine.
												transs := trans
												ms = m
												ns = n
												kls = kl
												kus = ku
												als = alpha
												as = aa.DeepCopy()
												xs = xx.DeepCopy()
												bls = beta
												ys = yy.DeepCopy()

												//                             Call the subroutine.
												if full {
													_ = Dgemv(trans, m, n, alpha, aa, xx, incx, beta, yy, incy)
												} else if banded {
													_ = Dgbmv(trans, m, n, kl, ku, alpha, aa, xx, incx, beta, yy, incy)
												}

												//                             Check if error-exit was taken incorrectly.
												if !ok {
													t.Fail()
													fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
													fatal = true
													goto label130
												}

												// //                             See what data changed inside subroutines.
												isame[0] = trans == transs
												isame[1] = ms == m
												isame[2] = ns == n
												if full {
													isame[3] = als == alpha
													isame[4] = reflect.DeepEqual(*aa, *as)
													isame[6] = reflect.DeepEqual(*xx, *xs)
													isame[8] = bls == beta
													if null {
														isame[9] = reflect.DeepEqual(*yy, *ys)
													} else {
														isame[9] = lderesV(mat.General, mat.Full, 1, ml, ys, yy, abs(incy), incy)
													}
												} else if banded {
													isame[3] = kls == kl
													isame[4] = kus == ku
													isame[5] = als == alpha
													isame[6] = reflect.DeepEqual(*aa, *as)
													isame[8] = reflect.DeepEqual(*xx, *xs)
													isame[10] = bls == beta
													if null {
														isame[11] = reflect.DeepEqual(*yy, *ys)
													} else {
														isame[11] = lderesV(mat.General, mat.Full, 1, ml, ys, yy, abs(incy), incy)
													}
												}

												//                             If data was incorrectly changed, report
												//                             and return.
												same = true
												for i = 1; i <= nargs; i++ {
													same = same && isame[i-1]
													if !isame[i-1] {
														t.Fail()
														fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
													}
												}
												if !same {
													fatal = true
													goto label130
												}

												if !null {
													//                                Check the result.
													dmvch(trans, m, n, alpha, a, nmax, x, sign(1, incx), beta, y, incy, yt, g, yy, eps, &err, &fatal, true, t)
													errmax = math.Max(errmax, err)
													//                                If got really bad answer, report and
													//                                return.
													if fatal {
														goto label130
													}
												} else {
													//                                Avoid repeating tests with M.le.0 or
													//                                N.le.0.
													goto label110
												}
											}
										}

									}
								}

							}

						}

					label110:
					}

				}
			}

			//     Report result.
			if errmax < thresh {
				if full {
					passL2(sname, nc, 3460*2, t)
				} else if banded {
					passL2(sname, nc, 13828*2, t)
				}
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label130:
			if full {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%3d,%4.1f, A,%3d, X,%3d,%4.1f, Y,%3d)         .\n", sname, nc, sname, trans.String(), m, n, alpha, lda, incx, beta, incy)
			} else if banded {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%3d,%3d,%3d,%4.1f, A,%3d, X,%3d,%4.1f, Y,%3d) .\n", sname, nc, sname, trans.String(), m, n, kl, ku, alpha, lda, incx, beta, incy)
			}

		} else if sname == "Dsymv" || sname == "Dsbmv" || sname == "Dspmv" {
			var full bool = sname[2] == 'y'
			var banded bool = sname[2] == 'b'
			var packed bool = sname[2] == 'p'

			if dchkeLevel2(sname); !common.infoc.ok {
				t.Fail()
			}

			//     Define the number of arguments.
			opts.Style = mat.Symmetric
			optsfull.Style = mat.Symmetric
			if full {
				nargs = 10
				opts.Storage = mat.Dense
			} else if banded {
				nargs = 11
				opts.Storage = mat.Banded
			} else if packed {
				nargs = 9
				opts.Storage = mat.Packed
				opts.Major = mat.Col
				optsfull.Major = mat.Col
			}

			nc = 0

			for _, maj := range []mat.MatMajor{mat.Row, mat.Col} {
				if packed && maj == mat.Row {
					continue
				}
				opts.Major = maj
				optsfull.Major = maj

				for _, n = range idim {
					if banded {
						nk = nkb
					} else {
						nk = 1
					}

					for ik = 1; ik <= nk; ik++ {
						if banded {
							k = kb[ik-1]
						} else {
							k = n - 1
						}
						opts.Kl, opts.Ku = k, k
						//           Set LDA to 1 more than minimum value if room.
						if banded {
							lda = k + 1
						} else {
							lda = n
						}
						if lda < nmax {
							lda++
						}
						//           Skip tests if not enough room.
						if lda > nmax {
							goto label1100
						}
						null = n <= 0

						for _, uplo = range ichu {
							opts.Uplo = uplo
							optsfull.Uplo = uplo
							//              Generate the matrix A.
							a = mf(nmax, nmax, optsfull)
							aa = mf(nmax, n, opts)
							dmakeL2M(n, n, a, nmax, aa, lda, k, k, &reset, 0.0)

							for _, incx = range inc {
								lx = abs(incx) * n

								//                 Generate the vector X.
								x = vf(n)
								xx = vf(lx)
								xiter := x.Iter(n, sign(1, incx))
								xxiter := xx.Iter(n, incx)
								dmakeL2V(1, n, x, sign(1, incx), xx, incx, 0, n-1, &reset, 0.5)
								if n > 1 {
									x.Set(xiter[n/2-1], 0)
									xx.Set(xxiter[n/2-1], 0)
								}

								for _, incy = range inc {
									ly = abs(incy) * n

									for _, alpha = range alf {

										for _, beta = range bet {
											//                          Generate the vector Y.
											y = vf(n)
											yy = vf(ly)
											dmakeL2V(1, n, y, sign(1, incy), yy, incy, 0, n-1, &reset, 0.0)

											nc++

											//                          Save every datum before calling the
											//                          subroutine.
											uplos = uplo
											ns = n
											ks = k
											als = alpha
											as = aa.DeepCopy()
											ldas = lda
											xs = xx.DeepCopy()
											bls = beta
											ys = yy.DeepCopy()

											//                          Call the subroutine.
											if full {
												_ = Dsymv(uplo, n, alpha, aa, xx, incx, beta, yy, incy)
											} else if banded {
												_ = Dsbmv(uplo, n, k, alpha, aa, xx, incx, beta, yy, incy)
											} else if packed {
												_ = Dspmv(uplo, n, alpha, aa.OffIdx(0).Vector(), xx, incx, beta, yy, incy)
											}

											//                          Check if error-exit was taken incorrectly.
											if !ok {
												t.Fail()
												fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
												fatal = true
												goto label1120
											}

											//                          See what data changed inside subroutines.
											isame[0] = uplo == uplos
											isame[1] = ns == n
											if full {
												isame[2] = als == alpha
												isame[3] = reflect.DeepEqual(aa, as)
												isame[4] = ldas == lda
												isame[5] = reflect.DeepEqual(xx, xs)
												isame[7] = bls == beta
												if null {
													isame[8] = reflect.DeepEqual(yy, ys)
												} else {
													isame[8] = lderesV(mat.General, mat.Full, 1, n, ys, yy, abs(incy), incy)
												}
											} else if banded {
												isame[2] = ks == k
												isame[3] = als == alpha
												isame[4] = reflect.DeepEqual(aa, as)
												isame[5] = ldas == lda
												isame[6] = reflect.DeepEqual(xx, xs)
												isame[8] = bls == beta
												if null {
													isame[9] = reflect.DeepEqual(yy, ys)
												} else {
													isame[9] = lderesV(mat.General, mat.Full, 1, n, ys, yy, abs(incy), incy)
												}
											} else if packed {
												isame[2] = als == alpha
												isame[3] = reflect.DeepEqual(aa, as)
												isame[4] = reflect.DeepEqual(xx, xs)
												isame[6] = bls == beta
												if null {
													isame[7] = reflect.DeepEqual(yy, ys)
												} else {
													isame[7] = lderesV(mat.General, mat.Full, 1, n, ys, yy, abs(incy), incy)
												}
											}

											//                          If data was incorrectly changed, report and
											//                          return.
											same = true
											for i = 1; i <= nargs; i++ {
												same = same && isame[i-1]
												if !isame[i-1] {
													t.Fail()
													fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
												}
											}
											if !same {
												fatal = true
												goto label1120
											}

											if !null {
												//                             Check the result.
												dmvch(NoTrans, n, n, alpha, a, nmax, x, sign(1, incx), beta, y, incy, yt, g, yy, eps, &err, &fatal, true, t)

												errmax = math.Max(errmax, err)

												//                             If got really bad answer, report and
												//                             return.
												if fatal {
													goto label1120
												}
											} else {
												//                             Avoid repeating tests with N.le.0
												goto label1110
											}

										}

									}
								}
							}
						}

					label1100:
					}

				label1110:
				}
			}

			//     Report result.
			if errmax < thresh {
				if full {
					passL2(sname, nc, 1441*2, t)
				} else if packed {
					passL2(sname, nc, 1441, t)
				} else if banded {
					passL2(sname, nc, 5761*2, t)
				}
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label1120:
			;
			if full {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%4.1f, A,%3d, X,%3d,%4.1f, Y,%3d)             .\n", sname, nc, sname, uplo.String(), n, alpha, lda, incx, beta, incy)
			} else if banded {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%3d,%4.1f, A,%3d, X,%3d,%4.1f, Y,%3d)         .\n", sname, nc, sname, uplo.String(), n, k, alpha, lda, incx, beta, incy)
			} else if packed {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%4.1f, AP, X,%3d,%4.1f, Y,%3d)                .\n", sname, nc, sname, uplo.String(), n, alpha, incx, beta, incy)
			}

		} else if sname == "Dtrmv" || sname == "Dtbmv" || sname == "Dtpmv" || sname == "Dtrsv" || sname == "Dtbsv" || sname == "Dtpsv" {
			var full bool = sname[2] == 'r'
			var banded bool = sname[2] == 'b'
			var packed bool = sname[2] == 'p'

			if dchkeLevel2(sname); !common.infoc.ok {
				t.Fail()
			}

			//     Define the number of arguments.
			opts.Style = mat.Triangular
			optsfull.Style = mat.Triangular
			if full {
				nargs = 8
				opts.Storage = mat.Dense
			} else if banded {
				nargs = 9
				opts.Storage = mat.Banded
			} else if packed {
				nargs = 7
				opts.Storage = mat.Packed
				opts.Major = mat.Col
				optsfull.Major = mat.Col
			}

			nc = 0
			for _, maj := range []mat.MatMajor{mat.Row, mat.Col} {
				if packed && maj == mat.Row {
					continue
				}
				opts.Major = maj
				optsfull.Major = maj

				for _, n = range idim {
					if banded {
						nk = nkb
					} else {
						nk = 1
					}

					for ik = 1; ik <= nk; ik++ {
						if banded {
							k = kb[ik-1]
						} else {
							k = n - 1
						}
						opts.Kl, opts.Ku = k, k
						//           Set LDA to 1 more than minimum value if room.
						if banded {
							lda = k + 1
						} else {
							lda = n
						}
						if lda < nmax {
							lda++
						}
						//           Skip tests if not enough room.
						if lda > nmax {
							goto label2100
						}
						null = n <= 0

						for _, uplo = range ichu {
							opts.Uplo = uplo
							optsfull.Uplo = uplo

							for _, trans = range icht {

								for _, diag = range ichd {
									opts.Diag = diag
									optsfull.Diag = diag

									//                    Generate the matrix A.
									a = mf(nmax, nmax, optsfull)
									aa = mf(nmax, nmax, opts)
									dmakeL2M(n, n, a, nmax, aa, lda, k, k, &reset, 0.0)

									for _, incx = range inc {
										lx = abs(incx) * n

										//                       Generate the vector X.
										x = vf(n)
										xx = vf(lx)
										xiter := x.Iter(n, sign(1, incx))
										xxiter := xx.Iter(n, incx)
										dmakeL2V(1, n, x, sign(1, incx), xx, incx, 0, n-1, &reset, 0.5)
										if n > 1 {
											x.Set(xiter[n/2-1], 0)
											xx.Set(xxiter[n/2-1], 0)
										}

										nc++

										//                       Save every datum before calling the subroutine.
										uplos = uplo
										transs = trans
										diags = diag
										ns = n
										ks = k
										as = aa.DeepCopy()
										ldas = lda
										xs = xx.DeepCopy()

										//                       Call the subroutine.
										if sname[3:5] == "mv" {
											if full {
												_ = Dtrmv(uplo, trans, diag, n, aa, xx, incx)
											} else if banded {
												_ = Dtbmv(uplo, trans, diag, n, k, aa, xx, incx)
											} else if packed {
												_ = Dtpmv(uplo, trans, diag, n, aa.OffIdx(0).Vector(), xx, incx)
											}
										} else if sname[3:5] == "sv" {
											if full {
												_ = Dtrsv(uplo, trans, diag, n, aa, xx, incx)
											} else if banded {
												_ = Dtbsv(uplo, trans, diag, n, k, aa, xx, incx)
											} else if packed {
												_ = Dtpsv(uplo, trans, diag, n, aa.OffIdx(0).Vector(), xx, incx)
											}
										}

										//                       Check if error-exit was taken incorrectly.
										if !ok {
											t.Fail()
											fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
											fatal = true
											goto label2120
										}

										//                       See what data changed inside subroutines.
										isame[0] = uplo == uplos
										isame[1] = trans == transs
										isame[2] = diag == diags
										isame[3] = ns == n
										if full {
											isame[4] = reflect.DeepEqual(aa, as)
											isame[5] = ldas == lda
											if null {
												isame[6] = reflect.DeepEqual(xx, xs)
											} else {
												isame[6] = lderesV(mat.General, mat.Full, 1, n, xs, xx, abs(incx), incx)
											}
										} else if banded {
											isame[4] = ks == k
											isame[5] = reflect.DeepEqual(aa, as)
											isame[6] = ldas == lda
											if null {
												isame[7] = reflect.DeepEqual(xx, xs)
											} else {
												isame[7] = lderesV(mat.General, mat.Full, 1, n, xs, xx, abs(incx), incx)
											}
										} else if packed {
											isame[4] = reflect.DeepEqual(aa, as)
											if null {
												isame[5] = reflect.DeepEqual(xx, xs)
											} else {
												isame[5] = lderesV(mat.General, mat.Full, 1, n, xs, xx, abs(incx), incx)
											}
										}

										//                       If data was incorrectly changed, report and
										//                       return.
										same = true
										for i = 1; i <= nargs; i++ {
											same = same && isame[i-1]
											if !isame[i-1] {
												t.Fail()
												fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
											}
										}
										if !same {
											fatal = true
											goto label2120
										}

										if !null {
											z := xs.DeepCopy()
											if sname[3:5] == "mv" {
												//                             Check the result.
												dmvch(trans, n, n, 1.0, a, nmax, xs, incx, 0.0, z, incx, xt, g, xx, eps, &err, &fatal, true, t)
											} else if sname[3:5] == "sv" {
												//                             Compute approximation to original vector.
												for i = 1; i <= n; i++ {
													z.Set(xxiter[i-1], xx.Get(xxiter[i-1]))
													xx.Set(xxiter[i-1], x.Get(xiter[i-1]))
												}
												dmvch(trans, n, n, 1.0, a, nmax, z, incx, 0.0, xs, incx, xt, g, xx, eps, &err, &fatal, true, t)
											}
											errmax = math.Max(errmax, err)
											//                          If got really bad answer, report and return.
											if fatal {
												goto label2120
											}
										} else {
											//                          Avoid repeating tests with N.le.0.
											goto label2110
										}

									}
								}

							}

						}

					label2100:
					}

				label2110:
				}
			}

			//     Report result.
			if errmax < thresh {
				if full {
					passL2(sname, nc, 241*2, t)
				} else if packed {
					passL2(sname, nc, 241, t)
				} else if banded {
					passL2(sname, nc, 961*2, t)
				}
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label2120:
			;
			if full {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%v,%v,%3d, A,%3d, X,%3d)                     .\n", sname, nc, sname, uplo.String(), trans.String(), diag.String(), n, lda, incx)
			} else if banded {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%v,%v,%3d,%3d, A,%3d, X,%3d)                 .\n", sname, nc, sname, uplo.String(), trans.String(), diag.String(), n, k, lda, incx)
			} else if packed {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%v,%v,%3d, AP, X,%3d)                        .\n", sname, nc, sname, uplo.String(), trans.String(), diag.String(), n, incx)
			}

		} else if sname == "Dger" {
			var nargs int = 9
			var nc int = 0

			opts.Style = mat.General
			opts.Storage = mat.Dense
			optsfull.Style = mat.General

			if dchkeLevel2(sname); !common.infoc.ok {
				t.Fail()
			}

			for i := range isame {
				isame[i] = true
			}

			for _, maj := range []mat.MatMajor{mat.Row, mat.Col} {
				opts.Major = maj
				optsfull.Major = maj

				for _, n = range idim {
					nd = n/2 + 1

					for im = 1; im <= 2; im++ {
						if im == 1 {
							m = max(n-nd, 0)
						}
						if im == 2 {
							m = min(n+nd, nmax)
						}
						//
						//           Set LDA to 1 more than minimum value if room.
						lda = m
						if lda < nmax {
							lda++
						}
						//           Skip tests if not enough room.
						if lda > nmax {
							continue
						}
						// laa = lda * n
						null = n <= 0 || m <= 0

						for _, incx = range inc {
							lx = abs(incx) * m

							//              Generate the vector X.
							x = vf(m)
							xx = vf(lx)
							xiter := x.Iter(m, sign(1, incx))
							xxiter := xx.Iter(m, incx)
							dmakeL2V(1, m, x, sign(1, incx), xx, incx, 0, m-1, &reset, 0.5)
							if m > 1 {
								x.Set(xiter[m/2-1], 0)
								xx.Set(xxiter[m/2-1], 0)
							}

							for _, incy = range inc {
								ly = abs(incy) * n

								//                 Generate the vector Y.
								y = vf(n)
								yy = vf(ly)
								yiter := y.Iter(n, sign(1, incy))
								yyiter := yy.Iter(n, incy)
								dmakeL2V(1, n, y, sign(1, incy), yy, incy, 0, n-1, &reset, 0.0)
								if n > 1 {
									y.Set(yiter[n/2-1], 0)
									yy.Set(yyiter[n/2-1], 0)
								}

								for _, alpha = range alf {
									//                    Generate the matrix A.
									a = mf(nmax, nmax, optsfull)
									aa = mf(lda, n, opts)
									dmakeL2M(m, n, a, nmax, aa, lda, m-1, n-1, &reset, 0.0)

									nc++

									//                    Save every datum before calling the subroutine.
									ms = m
									ns = n
									als = alpha
									as = aa.DeepCopy()
									ldas = lda
									xs = xx.DeepCopy()
									ys = yy.DeepCopy()

									//                    Call the subroutine.
									_ = Dger(m, n, alpha, xx, incx, yy, incy, aa)

									//                    Check if error-exit was taken incorrectly.
									if !ok {
										t.Fail()
										fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
										fatal = true
										goto label3140
									}

									//                    See what data changed inside subroutine.
									isame[0] = ms == m
									isame[1] = ns == n
									isame[2] = als == alpha
									isame[3] = reflect.DeepEqual(xx, xs)
									isame[4] = ldas == lda
									isame[5] = reflect.DeepEqual(yy, ys)
									if null {
										isame[7] = reflect.DeepEqual(aa, as)
									} else {
										isame[7] = lderesM(m, n, as, aa, lda)
									}

									//                    If data was incorrectly changed, report and return.
									same = true
									for i = 1; i <= nargs; i++ {
										same = same && isame[i-1]
										if !isame[i-1] {
											t.Fail()
											fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
										}
									}
									if !same {
										fatal = true
										goto label3140
									}

									if !null {
										a.ToColMajor()
										aa.ToColMajor()
										//                       Check the result column by column.
										z := mf(m, nmax, a.Opts.DeepCopy())
										w := y.DeepCopy()
										for i = 1; i <= m; i++ {
											z.Set(i-1, 0, x.Get(xiter[i-1]))
										}
										for j = 1; j <= n; j++ {
											if incy > 0 {
												w.Set(0, y.Get(j-1))
											} else {
												w.Set(0, y.Get(n-j))
											}
											dmvch(NoTrans, m, 1, alpha, z, nmax, w, 1, 1.0, a.Off(0, j-1).Vector(), 1, yt, g, aa.Off(0, j-1).Vector(), eps, &err, &fatal, true, t)
											errmax = math.Max(errmax, err)
											//                          If got really bad answer, report and return.
											if fatal {
												goto label3130
											}

										}
										a.ToRowMajor()
										aa.ToRowMajor()
									} else {
										//                       Avoid repeating tests with M.le.0 or N.le.0.
										goto label3110
									}

								}
							}
						}

					label3110:
					}

				}
			}

			//     Report result.
			if errmax < thresh {
				passL2(sname, nc, 388*2, t)
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label3130:
			;
			fmt.Printf("      THESE ARE THE RESULTS FOR COLUMN %3d\n", j)

		label3140:
			;
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%3d,%3d,%4.1f, X,%3d, Y,%3d, A,%3d)                  .\n", sname, nc, sname, m, n, alpha, incy, incx, lda)

		} else if sname == "Dsyr" || sname == "Dspr" {
			var full bool = sname[2] == 'y'
			var packed bool = sname[2] == 'p'

			if dchkeLevel2(sname); !common.infoc.ok {
				t.Fail()
			}

			//     Define the number of arguments.
			opts.Style = mat.Symmetric
			optsfull.Style = mat.Symmetric
			if full {
				nargs = 7
				opts.Storage = mat.Dense
			} else if packed {
				nargs = 6
				opts.Storage = mat.Packed
				opts.Major = mat.Col
				optsfull.Major = mat.Col
			}

			nc = 0

			for _, maj := range []mat.MatMajor{mat.Row, mat.Col} {
				if packed && maj == mat.Row {
					continue
				}
				opts.Major = maj
				optsfull.Major = maj

				for _, n = range idim {
					//        Set LDA to 1 more than minimum value if room.
					lda = n
					if lda < nmax {
						lda++
					}
					//        Skip tests if not enough room.
					if lda > nmax {
						goto label4100
					}

					for _, uplo = range ichu {
						upper = uplo == Upper
						opts.Uplo = uplo
						optsfull.Uplo = uplo

						for _, incx = range inc {
							lx = abs(incx) * n

							//              Generate the vector X.
							x = vf(n)
							xx = vf(lx)
							xiter := x.Iter(n, sign(1, incx))
							xxiter := xx.Iter(n, incx)
							dmakeL2V(1, n, x, sign(1, incx), xx, incx, 0, n-1, &reset, 0.5)
							if n > 1 {
								x.Set(xiter[n/2-1], 0)
								xx.Set(xxiter[n/2-1], 0)
							}

							for _, alpha = range alf {
								null = n <= 0 || alpha == 0

								//                 Generate the matrix A.
								a = mf(nmax, nmax, optsfull)
								aa = mf(lda, n, opts)

								nc++

								//                 Save every datum before calling the subroutine.
								uplos = uplo
								ns = n
								als = alpha
								as = aa.DeepCopy()
								ldas = lda
								xs := xx.DeepCopy()

								//                 Call the subroutine.
								if full {
									_ = Dsyr(uplo, n, alpha, xx, incx, aa)
								} else if packed {
									_ = Dspr(uplo, n, alpha, xx, incx, aa.OffIdx(0).Vector())
								}

								//                 Check if error-exit was taken incorrectly.
								if !ok {
									t.Fail()
									fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
									fatal = true
									goto label4120
								}

								//                 See what data changed inside subroutines.
								isame[0] = uplo == uplos
								isame[1] = ns == n
								isame[2] = als == alpha
								isame[3] = reflect.DeepEqual(xx, xs)
								if null {
									isame[5] = reflect.DeepEqual(aa, as)
								} else {
									isame[5] = lderesM(n, n, as, aa, lda)
								}
								if !packed {
									isame[6] = ldas == lda
								}

								//                 If data was incorrectly changed, report and return.
								same = true
								for i = 1; i <= nargs; i++ {
									same = same && isame[i-1]
									if !isame[i-1] {
										t.Fail()
										fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
									}
								}
								if !same {
									fatal = true
									goto label4120
								}

								if !null {
									a.ToColMajor()
									aa.ToColMajor()
									//                    Check the result column by column.
									z := mf(n, 1, opts)
									w := vf(n)
									for i = 1; i <= n; i++ {
										z.Set(i-1, 0, x.Get(xiter[i-1]))
									}
									ja = 1
									for j = 1; j <= n; j++ {
										w.Set(0, z.Get(j-1, 0))
										if upper {
											jj = 1
											lj = j
										} else {
											jj = j
											lj = n - j + 1
										}
										dmvch(NoTrans, lj, 1, alpha, z.OffIdx(jj-1), lj, w, 1, 1.0, a.Off(jj-1, j-1).Vector(), 1, yt, g, aa.OffIdx(ja-1).Vector(), eps, &err, &fatal, true, t)
										if full {
											if upper {
												ja += lda
											} else {
												ja += lda + 1
											}
										} else {
											ja += lj
										}
										errmax = math.Max(errmax, err)
										//                       If got really bad answer, report and return.
										if fatal {
											goto label4110
										}
									}
									a.ToRowMajor()
									aa.ToRowMajor()
								} else {
									//                    Avoid repeating tests if N.le.0.
									if n <= 0 {
										goto label4100
									}
								}

							}
						}

					}

				label4100:
				}
			}

			//     Report result.
			if errmax < thresh {
				if packed {
					passL2(sname, nc, 121, t)
				} else {
					passL2(sname, nc, 121*2, t)
				}
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label4110:
			;
			fmt.Printf("      THESE ARE THE RESULTS FOR COLUMN %3d\n", j)

		label4120:
			;
			if full {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%4.1f, X,%3d, A,%3d)                        .\n", sname, nc, sname, uplo.String(), n, alpha, incx, lda)
			} else if packed {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%4.1f, X,%3d, AP)                           .\n", sname, nc, sname, uplo.String(), n, alpha, incx)
			}

		} else if sname == "Dsyr2" || sname == "Dspr2" {
			var full = sname[2] == 'y'
			var packed = sname[2] == 'p'

			if dchkeLevel2(sname); !common.infoc.ok {
				t.Fail()
			}

			//     Define the number of arguments.
			opts.Style = mat.Symmetric
			optsfull.Style = mat.Symmetric
			if full {
				nargs = 9
				opts.Storage = mat.Dense
			} else if packed {
				nargs = 8
				opts.Storage = mat.Packed
				opts.Major = mat.Col
				optsfull.Major = mat.Col
			}

			nc = 0

			for _, maj := range []mat.MatMajor{mat.Row, mat.Col} {
				if packed && maj == mat.Row {
					continue
				}
				opts.Major = maj
				optsfull.Major = maj

				for _, n = range idim {
					//        Set LDA to 1 more than minimum value if room.
					lda = n
					if lda < nmax {
						lda++
					}
					//        Skip tests if not enough room.
					if lda > nmax {
						goto label5140
					}

					for _, uplo = range ichu {
						upper = uplo == Upper
						opts.Uplo = uplo
						optsfull.Uplo = uplo

						for _, incx = range inc {
							lx = abs(incx) * n

							//              Generate the vector X.
							x = vf(n)
							xx = vf(lx)
							xiter := x.Iter(n, sign(1, incx))
							xxiter := xx.Iter(n, incx)
							dmakeL2V(1, n, x, sign(1, incx), xx, incx, 0, n-1, &reset, 0.5)
							if n > 1 {
								x.Set(xiter[n/2-1], 0)
								xx.Set(xxiter[n/2-1], 0)
							}

							for _, incy = range inc {
								ly = abs(incy) * n

								//                 Generate the vector Y.
								y = vf(n)
								yy = vf(ly)
								yiter := y.Iter(n, sign(1, incy))
								yyiter := yy.Iter(n, incy)
								dmakeL2V(1, n, y, sign(1, incy), yy, incy, 0, n-1, &reset, 0.0)
								if n > 1 {
									y.Set(yiter[n/2-1], 0)
									yy.Set(yyiter[n/2-1], 0)
								}

								for _, alpha = range alf {
									null = n <= 0 || alpha == 0

									//                    Generate the matrix A.
									a = mf(nmax, nmax, optsfull)
									aa = mf(lda, n, opts)

									nc++

									//                    Save every datum before calling the subroutine.
									uplos = uplo
									ns = n
									als = alpha
									as = aa.DeepCopy()
									ldas = lda
									xs = xx.DeepCopy()
									ys = yy.DeepCopy()

									//                    Call the subroutine.
									if full {
										_ = Dsyr2(uplo, n, alpha, xx, incx, yy, incy, aa)
									} else if packed {
										_ = Dspr2(uplo, n, alpha, xx, incx, yy, incy, aa.OffIdx(0).Vector())
									}

									//                    Check if error-exit was taken incorrectly.
									if !ok {
										t.Fail()
										fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
										fatal = true
										goto label5160
									}

									//                    See what data changed inside subroutines.
									isame[0] = uplo == uplos
									isame[1] = ns == n
									isame[2] = als == alpha
									isame[3] = reflect.DeepEqual(xx, xs)
									isame[5] = reflect.DeepEqual(yy, ys)
									if null {
										isame[7] = reflect.DeepEqual(aa, as)
									} else {
										isame[7] = lderesM(n, n, as, aa, lda)
									}
									if !packed {
										isame[8] = ldas == lda
									}

									//                    If data was incorrectly changed, report and return.
									same = true
									for i = 1; i <= nargs; i++ {
										same = same && isame[i-1]
										if !isame[i-1] {
											t.Fail()
											fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
										}
									}
									if !same {
										fatal = true
										goto label5160
									}

									if !null {
										a.ToColMajor()
										aa.ToColMajor()
										//                    Check the result column by column.
										z := mf(n, 2, opts)
										w := vf(2)
										for i = 1; i <= n; i++ {
											z.Set(i-1, 0, x.Get(xiter[i-1]))
										}
										for i = 1; i <= n; i++ {
											z.Set(i-1, 1, y.Get(yiter[i-1]))
										}
										ja = 1
										for j = 1; j <= n; j++ {
											w.Set(0, z.Get(j-1, 1))
											w.Set(1, z.Get(j-1, 0))
											if upper {
												jj = 1
												lj = j
											} else {
												jj = j
												lj = n - j + 1
											}
											dmvch(NoTrans, lj, 2, alpha, z.Off(jj-1, 0), nmax, w, 1, 1.0, a.Off(jj-1, j-1).Vector(), 1, yt, g, aa.OffIdx(ja-1).Vector(), eps, &err, &fatal, true, t)
											if full {
												if upper {
													ja += lda
												} else {
													ja += lda + 1
												}
											} else {
												ja += lj
											}
											errmax = math.Max(errmax, err)
											//                          If got really bad answer, report and return.
											if fatal {
												goto label5150
											}
										}
										a.ToRowMajor()
										aa.ToRowMajor()
									} else {
										//                       Avoid repeating tests with N.le.0.
										if n <= 0 {
											goto label5140
										}
									}

								}
							}
						}

					}

				label5140:
				}
			}

			//     Report result.
			if errmax < thresh {
				if packed {
					passL2(sname, nc, 481, t)
				} else {
					passL2(sname, nc, 481*2, t)
				}
			} else {
				fmt.Printf(" %6s passed %6d computational tests\n ******* but with maximum test ratio %8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label5150:
			;
			fmt.Printf("      THESE ARE THE RESULTS FOR COLUMN %3d\n", j)

		label5160:

			if full {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%4.1f, X,%3d, Y,%3d, A,%3d)                  .\n", sname, nc, sname, uplo.String(), n, alpha, incx, incy, lda)
			} else if packed {
				fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n %6d: %6s(%v,%3d,%4.1f, X,%3d, Y,%3d, AP)                     .\n", sname, nc, sname, uplo.String(), n, alpha, incx, incy)
			}
		}
	}
}

func dmvch(trans mat.MatTrans, m, n int, alpha float64, a *mat.Matrix, nmax int, x *mat.Vector, incx int, beta float64, y *mat.Vector, incy int, yt, g, yy *mat.Vector, eps float64, err *float64, fatal *bool, mv bool, t *testing.T) {
	var tran bool
	var erri, one, zero float64
	var i, incxl, incyl, iy, j, jx, kx, ky, ml, nl int

	zero = 0.0
	one = 1.0

	tran = trans.IsTrans()
	if tran {
		ml = n
		nl = m
	} else {
		ml = m
		nl = n
	}
	if incx < 0 {
		kx = nl
		incxl = -1
	} else {
		kx = 1
		incxl = 1
	}
	if incy < 0 {
		ky = ml
		incyl = -1
	} else {
		ky = 1
		incyl = 1
	}
	xiter := x.Iter(nl, incx)
	yiter := yy.Iter(ml, incy)

	//     Compute expected result in YT using data in A, X and Y.
	//     Compute gauges in G.
	iy = ky
	for i = 1; i <= ml; i++ {
		yt.Set(yiter[i-1], zero)
		g.Set(yiter[i-1], zero)
		jx = kx
		if tran {
			for j = 1; j <= nl; j++ {
				yt.Set(yiter[i-1], yt.Get(yiter[i-1])+a.Get(j-1, i-1)*x.Get(xiter[j-1]))
				g.Set(yiter[i-1], g.Get(yiter[i-1])+math.Abs(a.Get(j-1, i-1)*x.Get(xiter[j-1])))
				jx += incxl
			}
		} else {
			for j = 1; j <= nl; j++ {
				yt.Set(yiter[i-1], yt.Get(yiter[i-1])+a.Get(i-1, j-1)*x.Get(xiter[j-1]))
				g.Set(yiter[i-1], g.Get(yiter[i-1])+math.Abs(a.Get(i-1, j-1)*x.Get(xiter[j-1])))
				jx += incxl
			}
		}
		yt.Set(yiter[i-1], alpha*yt.Get(yiter[i-1])+beta*y.Get(iy-1))
		g.Set(yiter[i-1], math.Abs(alpha)*g.Get(yiter[i-1])+math.Abs(beta*y.Get(iy-1)))
		iy += incyl
	}

	//     Compute the error ratio for this result.
	(*err) = zero
	for i = 1; i <= ml; i++ {
		// erri = math.Abs(yt.Get(i-1)-yy.Get((i-1)*abs(incy))) / eps
		erri = math.Abs(yt.Get(yiter[i-1])-yy.Get(yiter[i-1])) / eps
		if g.Get(yiter[i-1]) != zero {
			erri /= g.Get(yiter[i-1])
		}
		(*err) = math.Max(*err, erri)
		if (*err)*math.Sqrt(eps) >= one {
			goto label50
		}
	}
	//     If the loop completes, all results are at least half accurate.
	return

	//     Report fatal error.
label50:
	;
	*fatal = true
	fmt.Printf(" ******* FATAL ERROR - COMPUTED RESULT IS LESS THAN HALF ACCURATE *******\n           EXPECTED RESULT   COMPUTED RESULT\n")
	for i = 0; i < ml; i++ {
		if mv {
			fmt.Printf(" %7d%18.6f%18.6f\n", i, yt.Get(i), yy.Get(i))
		} else {
			fmt.Printf(" %7d%18.6f%18.6f\n", i, yy.Get(i), yt.Get(i))
		}
	}
}

func dmakeL2V(m, n int, a *mat.Vector, nmax int, aa *mat.Vector, lda, kl, ku int, reset *bool, transl float64) {
	var rogue, zero float64
	var i, j int

	zero = 0.0
	rogue = -1.0e10

	//     Generate data in array A.
	aiter := a.Iter(n*m, nmax)
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			if (i <= j && j-i <= ku) || (i >= j && i-j <= kl) {
				// a.Set(i-1+(j-1)*nmax, dbeg(reset)+transl)
				a.Set(aiter[i-1+(j-1)*m], dbeg(reset)+transl)
			} else {
				a.Set(i-1+(j-1)*nmax, zero)
			}
		}
	}

	//     Store elements in array AS in data structure required by routine.
	aa.SetAll(rogue)
	aaiter := aa.Iter(n*m, lda)
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			// aa.Set(i-1+(j-1)*lda, a.Get(i-1+(j-1)*nmax))
			aa.Set(aaiter[i-1+(j-1)*m], a.Get(aiter[i-1+(j-1)*m]))
		}
		// for i = m + 1; i <= abs(lda); i++ {
		// 	aa.Set(i-1+(j-1)*abs(lda), rogue)
		// }
	}
}

func dmakeL2M(m, n int, a *mat.Matrix, nmax int, aa *mat.Matrix, lda, kl, ku int, reset *bool, transl float64) {
	var one, rogue, zero float64
	var i, i1, i2, i3, ibeg, iend, ioff, j, kk int

	one = 1.0
	zero = 0.0
	rogue = -1.0e10

	gen := aa.Opts.Style == mat.General
	sym := aa.Opts.Style == mat.Symmetric
	tri := aa.Opts.Style == mat.Triangular
	upper := (sym || tri) && aa.Opts.Uplo == mat.Upper
	lower := (sym || tri) && aa.Opts.Uplo == mat.Lower
	unit := tri && aa.Opts.Diag == mat.Unit
	dense := aa.Opts.Storage == mat.Dense
	banded := aa.Opts.Storage == mat.Banded
	packed := aa.Opts.Storage == mat.Packed

	//     Generate data in array A.
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			if gen || (upper && i <= j) || (lower && i >= j) {
				if (i <= j && j-i <= ku) || (i >= j && i-j <= kl) {
					a.Set(i-1, j-1, dbeg(reset)+transl)
				} else {
					a.Set(i-1, j-1, zero)
				}
				if i != j {
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
	if gen && dense {
		for j = 1; j <= n; j++ {
			for i = 1; i <= m; i++ {
				aa.Set(i-1, j-1, a.Get(i-1, j-1))
			}
			for i = m + 1; i <= lda; i++ {
				aa.Set(i-1, j-1, rogue)
			}
		}
	} else if gen && banded {
		for j = 1; j <= n; j++ {
			for i1 = 1; i1 <= ku+1-j; i1++ {
				aa.Set(i1-1, j-1, rogue)
			}
			for i2 = i1; i2 <= min(kl+ku+1, ku+1+m-j); i2++ {
				aa.Set(i2-1, j-1, a.Get(i2+j-ku-1-1, j-1))
			}
			for i3 = i2; i3 <= lda; i3++ {
				aa.Set(i3-1, j-1, rogue)
			}
		}
	} else if (sym || tri) && dense {
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
	} else if (sym || tri) && banded {
		for j = 1; j <= n; j++ {
			if upper {
				kk = kl + 1
				ibeg = max(1, kl+2-j)
				if unit {
					iend = kl
				} else {
					iend = kl + 1
				}
			} else {
				kk = 1
				if unit {
					ibeg = 2
				} else {
					ibeg = 1
				}
				iend = min(kl+1, 1+m-j)
			}
			for i = 1; i <= ibeg-1; i++ {
				aa.Set(i-1, j-1, rogue)
			}
			for i = ibeg; i <= iend; i++ {
				aa.Set(i-1, j-1, a.Get(i+j-kk-1, j-1))
			}
			for i = iend + 1; i <= lda; i++ {
				aa.Set(i-1, j-1, rogue)
			}
		}
	} else if (sym || tri) && packed {
		ioff = 0
		for j = 1; j <= n; j++ {
			if upper {
				ibeg = 1
				iend = j
			} else {
				ibeg = j
				iend = n
			}
			for i = ibeg; i <= iend; i++ {
				ioff = ioff + 1
				aa.Set(ioff-1, 0, a.Get(i-1, j-1))
				if i == j {
					if unit {
						aa.Set(ioff-1, 0, rogue)
					}
				}
			}
		}
	}
}

func lderesV(_type mat.MatStyle, uplo mat.MatUplo, m, n int, aa, as *mat.Vector, lda, inc int) bool {
	var upper bool
	var i, ibeg, iend, j int

	upper = uplo == mat.Upper
	aiter := aa.Iter(n*m, inc)
	sort.Ints(aiter)
	if _type == mat.General {
		for j = 1; j <= n; j++ {
			// for i = m + 1; i <= lda; i++ {
			// 	if aa.Get(i-1+(j-1)*lda) != as.Get(i-1+(j-1)*lda) {
			// 		return false
			// 	}
			// }
			for i = 1; i <= lda; i++ {
				idx := i - 1 + (j-1)*lda
				if len(aiter) == 0 || idx != aiter[0] {
					if aa.Get(i-1+(j-1)*lda) != as.Get(i-1+(j-1)*lda) {
						return false
					}
				} else if idx == aiter[0] {
					aiter = aiter[1:]
				}
			}
		}
	} else if _type == mat.Symmetric {
		for j = 1; j <= n; j++ {
			if upper {
				ibeg = 1
				iend = j
			} else {
				ibeg = j
				iend = n
			}
			for i = 1; i <= ibeg-1; i++ {
				if aa.Get(i-1+(j-1)*lda) != as.Get(i-1+(j-1)*lda) {
					return false
				}
			}
			for i = iend + 1; i <= lda; i++ {
				if aa.Get(i-1+(j-1)*lda) != as.Get(i-1+(j-1)*lda) {
					return false
				}
			}
		}
	}

	return true
}

func lderesM(m, n int, aa, as *mat.Matrix, lda int) bool {
	var i, ibeg, iend, j int

	if aa.Opts.Style == mat.General && aa.Opts.Storage == mat.Dense {
		for j = 1; j <= n; j++ {
			for i = m + 1; i <= lda; i++ {
				if aa.Get(i-1, j-1) != as.Get(i-1, j-1) {
					return false
				}
			}
		}
	} else if aa.Opts.Style == mat.Symmetric && aa.Opts.Storage == mat.Dense {
		for j = 1; j <= n; j++ {
			if aa.Opts.Uplo == mat.Upper {
				ibeg = 1
				iend = j
			} else {
				ibeg = j
				iend = n
			}
			for i = 1; i <= ibeg-1; i++ {
				if aa.Get(i-1, j-1) != as.Get(i-1, j-1) {
					return false
				}
			}
			for i = iend + 1; i <= lda; i++ {
				if aa.Get(i-1, j-1) != as.Get(i-1, j-1) {
					return false
				}
			}
		}
	}

	return true
}

func dbeg(reset *bool) (dbegReturn float64) {
	var i *int = &common.begc.i
	var ic *int = &common.begc.ic
	var mi *int = &common.begc.mi

	if *reset {
		//        Initialize local variables.
		*mi = 891
		*i = 7
		*ic = 0
		(*reset) = false
	}
	//
	//     The sequence of values of I is bounded between 1 and 999.
	//     If initial I = 1,2,3,6,7 or 9, the period will be 50.
	//     If initial I = 4 or 8, the period will be 25.
	//     If initial I = 5, the period will be 10.
	//     IC is used to break up the period by skipping 1 value of I in 6.
	//
	*ic++
label10:
	;
	*i *= *mi
	*i -= 1000 * ((*i) / 1000)
	if *ic >= 5 {
		*ic = 0
		goto label10
	}
	dbegReturn = float64((*i)-500) / 1001.0
	return
}

func dchkeLevel2(srnamt string) {
	var alpha, beta float64
	var err error
	x := vf(1)
	y := vf(1)
	a := mf(1, 1, opts)

	//
	//  Tests the error exits from the Level 2 Blas.
	//  Requires a special version of the error-handling routine XERBLA.
	//  ALPHA, BETA, A, X and Y should not need to be defined.
	//
	//  Auxiliary routine for test program for Level 2 Blas.
	//
	//  -- Written on 10-August-1987.
	//     Richard Hanson, Sandia National Labs.
	//     Jeremy Du Croz, NAG Central Office.
	//
	errt := &common.infoc.errt
	ok := &common.infoc.ok
	lerr := &common.infoc.lerr

	*lerr = true

	switch srnamt {
	case "Dgemv":
		*ok = true
		*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		err = Dgemv('/', 0, 0, alpha, a, x, 1, beta, y, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Dgemv(NoTrans, -1, 0, alpha, a, x, 1, beta, y, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Dgemv(NoTrans, 0, -1, alpha, a, x, 1, beta, y, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		err = Dgemv(NoTrans, 2, 0, alpha, a, x, 1, beta, y, 1)
		Chkxer(srnamt, err)
	case "Dgbmv":
		*ok = true
		*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		err = Dgbmv('/', 0, 0, 0, 0, alpha, a, x, 1, beta, y, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Dgbmv(NoTrans, -1, 0, 0, 0, alpha, a, x, 1, beta, y, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Dgbmv(NoTrans, 0, -1, 0, 0, alpha, a, x, 1, beta, y, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("kl invalid: -1")
		err = Dgbmv(NoTrans, 0, 0, -1, 0, alpha, a, x, 1, beta, y, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ku invalid: -1")
		err = Dgbmv(NoTrans, 2, 0, 0, -1, alpha, a, x, 1, beta, y, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		err = Dgbmv(NoTrans, 0, 0, 1, 0, alpha, a, x, 1, beta, y, 1)
		Chkxer(srnamt, err)
		// case "Dsymv":
		// 	*ok = true
		// 	*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		// 	err = Dsymv('/', 0, alpha, a, x, beta, y)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("n invalid: -1")
		// 	err = Dsymv(Upper, -1, alpha, a, x, beta, y)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		// 	err = Dsymv(Upper, 2, alpha, a, x, beta, y)
		// 	Chkxer(srnamt, err)
		// case "Dsbmv":
		// 	*ok = true
		// 	*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		// 	err = Dsbmv('/', 0, 0, alpha, a, x, beta, y)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("n invalid: -1")
		// 	err = Dsbmv(Upper, -1, 0, alpha, a, x, beta, y)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("k invalid: -1")
		// 	err = Dsbmv(Upper, 0, -1, alpha, a, x, beta, y)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		// 	err = Dsbmv(Upper, 0, 1, alpha, a, x, beta, y)
		// 	Chkxer(srnamt, err)
		// case "Dspmv":
		// 	*ok = true
		// 	*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		// 	err = Dspmv('/', 0, alpha, a.OffIdx(0).Vector(), x, beta, y)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("n invalid: -1")
		// 	err = Dspmv(Upper, -1, alpha, a.OffIdx(0).Vector(), x, beta, y)
		// 	Chkxer(srnamt, err)
		// case "Dtrmv":
		// 	*ok = true
		// 	*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		// 	err = Dtrmv('/', NoTrans, NonUnit, 0, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		// 	err = Dtrmv(Upper, '/', NonUnit, 0, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("diag invalid: Unrecognized: /")
		// 	err = Dtrmv(Upper, NoTrans, '/', 0, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("n invalid: -1")
		// 	err = Dtrmv(Upper, NoTrans, NonUnit, -1, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		// 	err = Dtrmv(Upper, NoTrans, NonUnit, 2, a, x)
		// 	Chkxer(srnamt, err)
		// case "Dtbmv":
		// 	*ok = true
		// 	*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		// 	err = Dtbmv('/', NoTrans, NonUnit, 0, 0, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		// 	err = Dtbmv(Upper, '/', NonUnit, 0, 0, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("diag invalid: Unrecognized: /")
		// 	err = Dtbmv(Upper, NoTrans, '/', 0, 0, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("n invalid: -1")
		// 	err = Dtbmv(Upper, NoTrans, NonUnit, -1, 0, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("k invalid: -1")
		// 	err = Dtbmv(Upper, NoTrans, NonUnit, 0, -1, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		// 	err = Dtbmv(Upper, NoTrans, NonUnit, 0, 1, a, x)
		// 	Chkxer(srnamt, err)
		// case "Dtpmv":
		// 	*ok = true
		// 	*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		// 	err = Dtpmv('/', NoTrans, NonUnit, 0, a.OffIdx(0).Vector(), x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		// 	err = Dtpmv(Upper, '/', NonUnit, 0, a.OffIdx(0).Vector(), x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("diag invalid: Unrecognized: /")
		// 	err = Dtpmv(Upper, NoTrans, '/', 0, a.OffIdx(0).Vector(), x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("n invalid: -1")
		// 	err = Dtpmv(Upper, NoTrans, NonUnit, -1, a.OffIdx(0).Vector(), x)
		// 	Chkxer(srnamt, err)
		// case "Dtrsv":
		// 	*ok = true
		// 	*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		// 	err = Dtrsv('/', NoTrans, NonUnit, 0, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		// 	err = Dtrsv(Upper, '/', NonUnit, 0, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("diag invalid: Unrecognized: /")
		// 	err = Dtrsv(Upper, NoTrans, '/', 0, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("n invalid: -1")
		// 	err = Dtrsv(Upper, NoTrans, NonUnit, -1, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		// 	err = Dtrsv(Upper, NoTrans, NonUnit, 2, a, x)
		// 	Chkxer(srnamt, err)
		// case "Dtbsv":
		// 	*ok = true
		// 	*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		// 	err = Dtbsv('/', NoTrans, NonUnit, 0, 0, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		// 	err = Dtbsv(Upper, '/', NonUnit, 0, 0, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("diag invalid: Unrecognized: /")
		// 	err = Dtbsv(Upper, NoTrans, '/', 0, 0, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("n invalid: -1")
		// 	err = Dtbsv(Upper, NoTrans, NonUnit, -1, 0, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("k invalid: -1")
		// 	err = Dtbsv(Upper, NoTrans, NonUnit, 0, -1, a, x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		// 	err = Dtbsv(Upper, NoTrans, NonUnit, 0, 1, a, x)
		// 	Chkxer(srnamt, err)
		// case "Dtpsv":
		// 	*ok = true
		// 	*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		// 	err = Dtpsv('/', NoTrans, NonUnit, 0, a.OffIdx(0).Vector(), x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		// 	err = Dtpsv(Upper, '/', NonUnit, 0, a.OffIdx(0).Vector(), x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("diag invalid: Unrecognized: /")
		// 	err = Dtpsv(Upper, NoTrans, '/', 0, a.OffIdx(0).Vector(), x)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("n invalid: -1")
		// 	err = Dtpsv(Upper, NoTrans, NonUnit, -1, a.OffIdx(0).Vector(), x)
		// 	Chkxer(srnamt, err)
		// case "Dger":
		// 	*ok = true
		// 	*errt = fmt.Errorf("m invalid: -1")
		// 	err = Dger(-1, 0, alpha, x, y, a)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("n invalid: -1")
		// 	err = Dger(0, -1, alpha, x, y, a)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		// 	err = Dger(2, 0, alpha, x, y, a)
		// 	Chkxer(srnamt, err)
		// case "Dsyr":
		// 	*ok = true
		// 	*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		// 	err = Dsyr('/', 0, alpha, x, a)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("n invalid: -1")
		// 	err = Dsyr(Upper, -1, alpha, x, a)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		// 	err = Dsyr(Upper, 2, alpha, x, a)
		// 	Chkxer(srnamt, err)
		// case "Dspr":
		// 	*ok = true
		// 	*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		// 	err = Dspr('/', 0, alpha, x, a.OffIdx(0)).Vector()
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("n invalid: -1")
		// 	err = Dspr(Upper, -1, alpha, x, a.OffIdx(0)).Vector()
		// 	Chkxer(srnamt, err)
		// case "Dsyr2":
		// 	*ok = true
		// 	*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		// 	err = Dsyr2('/', 0, alpha, x, y, a)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("n invalid: -1")
		// 	err = Dsyr2(Upper, -1, alpha, x, y, a)
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		// 	err = Dsyr2(Upper, 2, alpha, x, y, a)
		// 	Chkxer(srnamt, err)
		// case "Dspr2":
		// 	*ok = true
		// 	*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		// 	err = Dspr2('/', 0, alpha, x, y, a.OffIdx(0)).Vector()
		// 	Chkxer(srnamt, err)
		// 	*errt = fmt.Errorf("n invalid: -1")
		// 	err = Dspr2(Upper, -1, alpha, x, y, a.OffIdx(0)).Vector()
		// 	Chkxer(srnamt, err)
	}

	if *ok {
		fmt.Printf(" %6s passed the tests of error-exits\n", srnamt)
	} else {
		fmt.Printf(" ******* %6s FAILED THE TESTS OF ERROR-EXITS *******\n", srnamt)
	}
}

// func BenchmarkDgemv(b *testing.B) {
// 	var reset, tran bool
// 	var alpha, beta float64
// 	var i, j, kl, ku, lda, m, ml, n, nd, nl int
// 	var a, aa *mat.Matrix
// 	var x, xx, y, yy *mat.Vector
// 	n = 10000
// 	incx := 1
// 	incy := 1
// 	optsfull := opts.DeepCopy()
// 	a = mf(n, n, opts)
// 	x = vf(n)
// 	y = vf(n)
// 	yy = vf(n)

// 	for j = 1; j <= n; j++ {
// 		for i = 1; i <= n; i++ {
// 			a.Set(i-1, j-1, float64(max(i-j+1, 0)))
// 		}
// 		x.Set(j-1, float64(j))
// 		y.Set(j-1, 0)
// 	}
// 	for j = 1; j <= n; j++ {
// 		yy.Set(j-1, float64(j*((j+1)*j))/2-float64(((j+1)*j*(j-1)))/3)
// 	}
// 	//     YY holds the exact result. On exit from SMVCH YT holds
// 	//     the result computed by SMVCH.
// 	reset = true

// 	opts.Style = mat.General
// 	optsfull.Style = mat.General
// 	opts.Storage = mat.Dense
// 	opts.Major = mat.Row
// 	optsfull.Major = mat.Row

// 	nd = n/2 + 1

// 	m = min(n+nd, n)

// 	ku = n - 1
// 	kl = m - 1

// 	//              Set LDA to 1 more than minimum value if room.
// 	lda = m

// 	//              Generate the matrix A.
// 	a = mf(n, n, optsfull)
// 	aa = mf(lda, n, opts)
// 	dmakeL2M(m, n, a, n, aa, lda, kl, ku, &reset, 0.0)

// 	trans := NoTrans
// 	tran = trans.IsTrans()

// 	if tran {
// 		ml = n
// 		nl = m
// 	} else {
// 		ml = m
// 		nl = n
// 	}

// 	lx := abs(incx) * nl

// 	//                    Generate the vector X.
// 	x = vf(nl, incx)
// 	xx = vf(lx, incx)
// 	dmakeL2V(1, nl, x, 1, xx, abs(incx), 0, nl-1, &reset, 0.5)
// 	if nl > 1 {
// 		x.Set(nl/2-1, 0)
// 		xx.Set(abs(incx)*(nl/2-1), 0)
// 	}

// 	ly := abs(incy) * ml

// 	//                             Generate the vector Y.
// 	y = vf(ml, incy)
// 	yy = vf(ly, incy)
// 	dmakeL2V(1, ml, y, 1, yy, abs(incy), 0, ml-1, &reset, 0.0)

// 	b.ResetTimer()
// 	for i := 0; i < b.N; i++ {
// 		_ = Dgemv(trans, m, n, alpha, aa, xx, beta, yy)
// 	}

// }
